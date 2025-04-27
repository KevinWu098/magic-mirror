import time
import os
import json
from PIL import Image, ImageDraw, ImageColor
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
import re
import traceback # Added for detailed error logging


# --- Gemini Initialization ---
def initialize_gemini_client():
    """Loads API key and initializes the Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        traceback.print_exc()
        return None

# --- Helper Function for Gemini API Call ---
def call_gemini_api(client, model_name, object_name, PIL_image):
    """Call the Gemini API using refined prompting for bounding boxes."""
    if not client:
        raise ValueError("Gemini client is not initialized.")

    start = time.time()

    # Refined System Instruction (Ensure pure JSON output)
    bounding_box_system_instructions = (
        "You are an expert object detection assistant. "
        "Your task is to identify all instances of the requested object(s) in the provided image. "
        "Respond ONLY with a valid JSON dictionary containing a single key 'objects'. The value associated with 'objects' must be a JSON list of detected object dictionaries. "
        "Do not include any other text, explanations, or markdown formatting like ```json. "
        "Each object dictionary in the list must have exactly two keys: "
        "1. 'label': A descriptive string identifying the object instance (e.g., 'person 1', 'person 2'). "
        "2. 'box_2d': A list of four integers representing the bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates relative to the top-left corner of the image (0 to image dimension)."
        "Ensure the final JSON output is syntactically correct and coordinates are within image bounds. "
        'For example: {"objects": [{"label": "person 1", "box_2d": [100, 150, 300, 450]}]}'
    )

    # Refined User Prompt
    prompt = f"Detect all instances of '{object_name}'. Follow the JSON output format specified in the system instructions precisely. Provide bounding boxes in pixel coordinates [x_min, y_min, x_max, y_max]."

    # Define Config
    config = types.GenerateContentConfig(
        # system_instruction=bounding_box_system_instructions, # System prompt often works better directly in contents for multimodal
        temperature=0.2, # Lower temperature for more deterministic bounding boxes
        # Adjust safety settings if needed, blocking too aggressively can be an issue
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
        response_mime_type="application/json",
    )

    print(f"Sending prompt to Gemini for '{object_name}'...")
    try:
        # Combine system instructions with the prompt for potentially better adherence
        combined_content = [
            bounding_box_system_instructions, # System prompt first
            prompt,
            PIL_image
        ]
        response = client.models.generate_content(
            model=model_name,
            contents=combined_content,
            generation_config=config, # Use generation_config for newer versions
        )

        # --- Response Handling ---
        response_text = ""
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            response_text = response.candidates[0].content.parts[0].text
            # Check finish reason if available
            finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            if finish_reason and finish_reason != 1: # 1 is typically "STOP" or normal completion
                 print(f"Warning: Gemini response finished with reason: {finish_reason}")
                 # Handle specific reasons like SAFETY if needed
                 if finish_reason == 3: # 3 is often SAFETY
                     raise ValueError(f"Gemini response blocked due to safety ({finish_reason}). Check prompt/image.")

        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            block_reason_message = getattr(response.prompt_feedback, 'block_reason_message', '')
            print(f"Gemini response was blocked. Reason: {block_reason}. Message: {block_reason_message}")
            raise ValueError(f"Gemini response blocked: {block_reason}")
        else:
            # General case for empty or unexpected response
            print(f"Gemini response was empty or unexpected: {response}")
            # Look for safety ratings if available on the top-level response object
            safety_ratings = getattr(response, 'prompt_feedback', None)
            if safety_ratings and safety_ratings.safety_ratings:
                 print("Safety Ratings:", safety_ratings.safety_ratings)

            raise ValueError("Gemini response empty or unexpected")

        # print(f"'{object_name}' Gemini raw response: {response_text}") # Debugging: keep if needed
        print(f"Gemini API Call function took {time.time() - start:.2f} seconds")
        return response_text

    except Exception as e:
        print(f"Error during Gemini API call function: {e}")
        traceback.print_exc()
        raise e # Re-raise the exception


# --- Helper Function for Parsing Gemini Response ---
def parse_gemini_response(response_text, image_width, image_height):
    """Parses the JSON response from Gemini to extract bounding boxes."""
    start_parse = time.time()
    bounding_boxes = []

    # 1. Strip potential markdown fences (though less likely with refined prompt & mime type)
    response_json_text = response_text.strip()
    if response_json_text.startswith("```json"):
        response_json_text = response_json_text[7:]
    if response_json_text.endswith("```"):
        response_json_text = response_json_text[:-3]
    response_json_text = response_json_text.strip()

    if not response_json_text:
        print("Error: Received empty response text after stripping.")
        return [] # Return empty list if no text

    # 2. Parse JSON
    try:
        outer_data = json.loads(response_json_text)
        if not isinstance(outer_data, dict):
            raise ValueError("Expected a dictionary at the top level.")
        if "objects" not in outer_data or not isinstance(outer_data["objects"], list):
            raise ValueError("Expected a list associated with the key 'objects'.")
        parsed_data = outer_data["objects"]

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")
        print(f"Received text: >>>
{response_json_text}
<<<")
        traceback.print_exc()
        return [] # Return empty list on parse failure
    except ValueError as e:
        print(f"Error: Unexpected JSON structure: {e}")
        print(f"Received JSON data: {outer_data}")
        traceback.print_exc()
        return [] # Return empty list on structure failure

    # 3. Process parsed JSON data (expecting pixel coordinates)
    for item in parsed_data:
        if not isinstance(item, dict) or not all(key in item for key in ["label", "box_2d"]):
            print(f"Warning: Skipping item with missing/invalid keys: {item}")
            continue

        try:
            label = item["label"]
            # Expecting [x_min, y_min, x_max, y_max] in pixel coordinates
            box_coords_pix = item["box_2d"]

            if not isinstance(box_coords_pix, list) or len(box_coords_pix) != 4:
                print(f"Warning: Skipping item '{label}' with invalid box_2d format: {box_coords_pix}")
                continue

            # Validate and clamp coordinates
            x1, y1, x2, y2 = map(int, box_coords_pix)
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(x1 + 1, min(x2, image_width)) # Ensure x2 > x1
            y2 = max(y1 + 1, min(y2, image_height)) # Ensure y2 > y1


            # Ensure calculated pixel coordinates are valid (x1 < x2 and y1 < y2)
            if x1 < x2 and y1 < y2:
                # Store the validated pixel coordinates in [x1, y1, x2, y2] order
                bounding_boxes.append(
                    {"box_2d": [x1, y1, x2, y2], "label": label}
                )
                # print(f"Storing box: Label='{label}', Coords(pixel XYxy)= [ {x1}, {y1}, {x2}, {y2} ]") # Debugging: keep if needed
            else:
                print(f"Warning: Skipping parsed box with invalid coordinates after clamping for label '{label}': Original=({box_coords_pix}) Clamped=({x1},{y1},{x2},{y2}) Image=({image_width}x{image_height})")
                continue

        except (ValueError, TypeError, KeyError) as e:
            print(f"Warning: Could not process item {item} - {e}")
            traceback.print_exc()


    print(f"Response Parsing section took {time.time() - start_parse:.2f} seconds")
    print(f"Found {len(bounding_boxes)} valid bounding boxes.")
    return bounding_boxes


# --- Drawing Function (Optional - For testing) ---
def draw_bounding_boxes_pil(PIL_image, bounding_boxes, output_path="output_gemini_boxes.jpg"):
    """Draws bounding boxes onto a PIL image."""
    if not bounding_boxes:
        print("No bounding boxes provided to draw.")
        return

    start_draw = time.time()
    draw_image = PIL_image.copy() # Work on a copy
    draw = ImageDraw.Draw(draw_image)
    box_count = 0
    colors = list(ImageColor.colormap.keys())

    for i, bounding_box_info in enumerate(bounding_boxes):
        label = bounding_box_info["label"]
        x1, y1, x2, y2 = bounding_box_info["box_2d"] # Already pixel coords

        bounding_box_color = colors[i % len(colors)]
        bounding_box_thickness = 2
        font_size = 15
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(font_size)
        except Exception:
            font = None

        draw.rectangle(
            ((x1, y1), (x2, y2)),
            outline=bounding_box_color,
            width=bounding_box_thickness,
        )

        text_position = (x1 + 4, y1 + 2)
        if font:
            try:
                text_bbox = draw.textbbox(text_position, label, font=font)
                bg_rect = (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1)
                draw.rectangle(bg_rect, fill=bounding_box_color)
                draw.text(text_position, label, fill="white", font=font)
            except AttributeError: # Fallback for older PIL versions
                draw.text(text_position, label, fill=bounding_box_color, font=font)
        else:
            draw.text(text_position, label, fill=bounding_box_color)

        # print(f"Drew box for '{label}' with color {bounding_box_color}: ({x1}, {y1}) to ({x2}, {y2})") # Debugging: keep if needed
        box_count += 1

    if box_count > 0:
        draw_image.save(output_path)
        print(f"{box_count} Bounding box(es) drawn and saved to {output_path}")
    else:
        print("No valid bounding boxes found to draw.")

    print(f"Draw Bounding Boxes section took {time.time() - start_draw:.2f} seconds")


# --- Main Execution Logic (Example Usage) ---
def main():
    start_total = time.time()

    # --- Inputs ---
    start_load = time.time()
    image_path = "people.jpg" # Make sure this image exists
    object_name = "person" # More specific object
    try:
        PIL_image = Image.open(image_path).convert("RGB") # Ensure RGB
        width, height = PIL_image.size
        print(f"Image '{image_path}' loaded ({width}x{height}). Took {time.time() - start_load:.2f} seconds")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        traceback.print_exc()
        return

    # --- Gemini Initialization ---
    start_gemini_init = time.time()
    client = initialize_gemini_client()
    if not client:
        return # Exit if client failed to initialize
    # Select a recent model, check availability: https://ai.google.dev/gemini-api/docs/models/gemini
    # model_name = "gemini-1.5-flash-latest" # Use latest flash model
    # Or specify a version if needed:
    model_name = "gemini-1.5-flash-001"
    print(f"Using Gemini model: {model_name}")
    print(f"Gemini Initialization section took {time.time() - start_gemini_init:.2f} seconds")


    # --- Gemini API Call & Parsing ---
    try:
        response_text = call_gemini_api(client, model_name, object_name, PIL_image)
        bounding_boxes = parse_gemini_response(response_text, width, height)

        if not bounding_boxes:
            print("Error: No valid bounding boxes were parsed from the Gemini response.")
            return

        # --- Draw Bounding Boxes (Optional: For visual verification) ---
        draw_bounding_boxes_pil(PIL_image, bounding_boxes, output_path="output_gemini_boxes_main.jpg")

    except ValueError as e: # Catch specific errors from API call/parsing
        print(f"Failed processing Gemini response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    print(f"Total execution time: {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    # Example execution when script is run directly
    main()
# --- Remove old main execution logic remnants ---
# Delete lines 83-314 which contained the old main logic mixed with helpers
# The new main function above provides a clean example.
# The core logic is now in initialize_gemini_client, call_gemini_api, parse_gemini_response.

# Remove CUDA optimization section as it's less relevant here and potentially confusing
# Remove lines 111-121

# Remove unused imports if any (e.g., torch if only used for optimizations)
# Keep numpy as it might be useful elsewhere, PIL, os, json, time, re, dotenv, google.genai are used.
