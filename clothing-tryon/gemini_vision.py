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


# --- Helper Function for Gemini API Call ---
def call_gemini_api(client, model_name, object_name, PIL_image):
    """Call the Gemini API using refined prompting for bounding boxes."""
    start = time.time()

    # Refined System Instruction
    bounding_box_system_instructions = (
        "You are an expert object detection assistant. "
        "Your task is to identify all instances of the requested object(s) in the provided image. "
        "Respond ONLY with a valid JSON dictionary containing a single key 'objects'. The value associated with 'objects' must be a JSON list of detected object dictionaries. "
        "Do not include any other text, explanations, or markdown formatting like ```json. "
        "Each object dictionary in the list must have exactly two keys: "
        "1. 'label': A descriptive string identifying the object instance (e.g., 'woman in red shirt', 'person 3'). "
        "2. 'box_2d': A list of four integers representing the bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates relative to the top-left corner of the image."
        "Ensure the final JSON output is syntactically correct. "
        'For example, a valid response structure is: {"objects": [{"label": "example object", "box_2d": [10, 20, 100, 120]}]}.'
    )

    # Refined User Prompt
    prompt = f"Detect all instances of '{object_name}'. Follow the JSON output format specified in the system instructions precisely."

    # Define Config (including refined system instruction and potentially other params)
    # Note: Safety settings might need to be added here if GenerateContentConfig supports it in v0.8.5
    config = types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,  # Keep temperature for some variability if needed
        # safety_settings=safety_settings, # Add safety settings list here if needed/supported
        response_mime_type="application/json",  # Explicitly request JSON MIME type
    )

    print(f"Sending prompt to Gemini: '{prompt}'")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, PIL_image],
            config=config,  # Pass the config object using the 'config' parameter name
        )

        # --- Response Handling ---
        response_text = ""
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            response_text = response.candidates[0].content.parts[0].text
        # Add check for finish_reason if available and useful (e.g., SAFETY)
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            print(f"Gemini response was blocked. Reason: {block_reason}")
            raise ValueError(f"Gemini response blocked: {block_reason}")
        else:
            # General case for empty or unexpected response
            print(f"Gemini response was empty or unexpected: {response}")
            raise ValueError("Gemini response empty or unexpected")

        print(f"{object_name} Gemini raw response: {response_text}")
        print(f"Gemini API Call function took {time.time() - start:.2f} seconds")
        return response_text

    except Exception as e:
        print(f"Error during Gemini API call function: {e}")
        # Consider logging the full exception traceback for debugging
        # import traceback
        # traceback.print_exc()
        raise e


# --- Main Execution Logic ---
def main():
    start_total = time.time()

    # --- Inputs ---
    # ... (Input loading remains the same) ...
    start_load = time.time()
    image_path = "people.jpg"
    object_name = "all the people in this image"
    try:
        PIL_image = Image.open(image_path)
        width, height = PIL_image.size
        if PIL_image.mode not in ("RGB", "L"):
            print(f"Converting image mode from {PIL_image.mode} to RGB")
            PIL_image = PIL_image.convert("RGB")
        elif PIL_image.mode == "L":
            print(f"Converting image mode from {PIL_image.mode} to RGB")
            PIL_image = PIL_image.convert("RGB")
        np_image = np.array(PIL_image)
        print(f"Image Loading section took {time.time() - start_load:.2f} seconds")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # --- Inference Optimizations (Optional) ---
    start_optim = time.time()
    try:
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(
            f"Inference Optimization section took {time.time() - start_optim:.2f} seconds"
        )
    except Exception as e:
        print(f"CUDA optimization failed (continuing without it): {e}")
        if torch.is_autocast_enabled():
            torch.autocast("cuda").__exit__(None, None, None)

    # --- Gemini Initialization ---
    start_gemini_init = time.time()
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.0-flash"
        # Define instructions & settings, though they aren't passed in the minimal call
        bounding_box_system_instructions = (
            "Return bounding boxes as a JSON array of objects..."
        )
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            # ... (other settings)
        ]
        print(
            f"Gemini Initialization section took {time.time() - start_gemini_init:.2f} seconds"
        )
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return

    # --- Gemini API Call (using the helper function) ---
    try:
        # Call the updated helper function (removed unused args)
        response_text = call_gemini_api(client, model_name, object_name, PIL_image)

        # --- Parse Response (Expecting new JSON structure) ---
        start_parse = time.time()
        bounding_boxes = []

        # 1. Strip potential markdown fences (though less likely with refined prompt)
        response_json_text = response_text.strip()
        if response_json_text.startswith("```json"):
            response_json_text = response_json_text[7:]
        if response_json_text.endswith("```"):
            response_json_text = response_json_text[:-3]
        response_json_text = response_json_text.strip()

        # 2. Parse JSON
        try:
            # Load the entire JSON response (expecting a dictionary)
            outer_data = json.loads(response_json_text)
            if not isinstance(outer_data, dict):
                raise ValueError(
                    "Expected a dictionary at the top level of JSON response."
                )
            # Extract the list from the 'objects' key
            if "objects" not in outer_data or not isinstance(
                outer_data["objects"], list
            ):
                raise ValueError(
                    "Expected a list associated with the key 'objects' in JSON response."
                )
            parsed_data = outer_data["objects"]  # This is the list we want to process

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON: {e}")
            print(f"Received text: {response_json_text}")
            return
        except ValueError as e:
            print(f"Error: Unexpected JSON structure: {e}")
            return

        # 3. Process parsed JSON data (adapted for new structure)
        for item in parsed_data:
            if not all(key in item for key in ["label", "box_2d"]):
                print(
                    f"Warning: Skipping item with missing keys ('label' or 'box_2d'): {item}"
                )
                continue

            try:
                label = item["label"]
                box_coords_scaled = item[
                    "box_2d"
                ]  # Assume these are scaled [y1,x1,y2,x2]

                if (
                    not isinstance(box_coords_scaled, list)
                    or len(box_coords_scaled) != 4
                ):
                    print(
                        f"Warning: Skipping item '{label}' with invalid box_2d format: {box_coords_scaled}"
                    )
                    continue

                # Extract scaled coordinates assuming [y_min_scaled, x_min_scaled, y_max_scaled, x_max_scaled]
                y_min_scaled, x_min_scaled, y_max_scaled, x_max_scaled = map(
                    int, box_coords_scaled
                )

                # Unscale to pixel coordinates (like original draw_bounding_box)
                y1_pix = int(y_min_scaled / 1000 * height)
                x1_pix = int(x_min_scaled / 1000 * width)
                y2_pix = int(y_max_scaled / 1000 * height)
                x2_pix = int(x_max_scaled / 1000 * width)

                # Ensure calculated pixel coordinates are valid
                if x1_pix < x2_pix and y1_pix < y2_pix:
                    # Store the calculated *pixel* coordinates in [x1, y1, x2, y2] order
                    bounding_boxes.append(
                        {"box_2d": [x1_pix, y1_pix, x2_pix, y2_pix], "label": label}
                    )
                    # Update print statement
                    print(
                        f"Storing box: Label='{label}', Coords(scaled YXyx)= [ {y_min_scaled}, {x_min_scaled}, {y_max_scaled}, {x_max_scaled} ], Coords(pixel XYxy)= [ {x1_pix}, {y1_pix}, {x2_pix}, {y2_pix} ]"
                    )
                else:
                    print(
                        f"Warning: Skipping parsed box with invalid coordinates for label '{label}': Scaled=({y_min_scaled},{x_min_scaled},{y_max_scaled},{x_max_scaled}) Pixel=({x1_pix},{y1_pix},{x2_pix},{y2_pix}) "
                    )
                    continue

            except (ValueError, TypeError, KeyError) as e:
                print(f"Warning: Could not process item {item} - {e}")

        # Check if any boxes were successfully parsed and stored
        if not bounding_boxes:
            print(
                "Error: No valid bounding boxes could be processed from the JSON response."
            )
            return
        print(f"Response Parsing section took {time.time() - start_parse:.2f} seconds")

        # --- Draw Bounding Boxes (using PIL and direct pixel coords) ---
        start_draw = time.time()
        output_path = "bounding_box_test_pil.jpg"
        # Use a copy if you want to keep the original PIL_image unmodified
        # draw_image = PIL_image.copy()
        # draw = ImageDraw.Draw(draw_image)
        # Or draw directly on PIL_image if modification is okay:
        draw = ImageDraw.Draw(PIL_image)
        box_count = 0
        colors = list(ImageColor.colormap.keys())

        # Iterate through the stored bounding_boxes list (now containing pixel coords)
        for i, bounding_box in enumerate(bounding_boxes):
            label = bounding_box["label"]
            # Get the stored *pixel* coordinates
            x1, y1, x2, y2 = bounding_box["box_2d"]

            # Use pixel coordinates directly for drawing
            bounding_box_color = colors[i % len(colors)]
            bounding_box_thickness = 2
            font_size = 15
            try:
                from PIL import ImageFont

                font = ImageFont.load_default(font_size)
            except Exception:
                font = None  # Fallback

            # Draw rectangle using direct pixel coordinates
            draw.rectangle(
                ((x1, y1), (x2, y2)),
                outline=bounding_box_color,
                width=bounding_box_thickness,
            )

            # Add label using direct pixel coordinates
            text_position = (x1 + 4, y1 + 2)
            if font:
                try:
                    text_bbox = draw.textbbox(text_position, label, font=font)
                    # Adjust background rect slightly
                    bg_rect = (
                        text_bbox[0] - 2,
                        text_bbox[1] - 1,
                        text_bbox[2] + 2,
                        text_bbox[3] + 1,
                    )
                    draw.rectangle(bg_rect, fill=bounding_box_color)
                    draw.text(text_position, label, fill="white", font=font)
                except AttributeError:
                    draw.text(text_position, label, fill=bounding_box_color, font=font)
            else:
                draw.text(text_position, label, fill=bounding_box_color)

            print(
                f"Drew box for '{label}' with color {bounding_box_color}: ({x1}, {y1}) to ({x2}, {y2})"
            )
            box_count += 1

        if box_count > 0:
            # Save the modified PIL image (or draw_image if using a copy)
            PIL_image.save(output_path)
            print(f"{box_count} Bounding box(es) drawn and saved to {output_path}")
        else:
            print(
                "No valid bounding boxes found in the response to draw."
            )  # Should not happen if parsing succeeded

        print(
            f"Draw Bounding Boxes section took {time.time() - start_draw:.2f} seconds"
        )

    except ValueError as e:
        print(f"Failed to get valid response from Gemini: {e}")
    except Exception as e:
        print(f"An unexpected error occurred after Gemini initialization: {e}")

    # --- Exit Autocast ---
    try:
        if torch.is_autocast_enabled():
            torch.autocast("cuda").__exit__(None, None, None)
    except Exception as e:
        print(f"Error exiting autocast: {e}")

    print(f"Total execution time: {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    main()
