from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import os
import io
import sys
from PIL import Image, ImageDraw, ImageColor
import numpy as np
import torch
import random
import traceback
import time
import math
import base64
import threading
import concurrent.futures
from transformers import CLIPVisionModelWithProjection
from ultralytics import FastSAM, YOLO
import cv2
import uuid
import requests
from dotenv import load_dotenv

# --- Configuration ---
MODEL_PATH = "/workspace/FitDiT/local_model_dir"
DEVICE = "cuda:0"
OFFLOAD = False
AGGRESSIVE_OFFLOAD = False
WITH_FP16 = True
GEMINI_MODEL_NAME = "gemini-1.5-flash-001"

# Load environment variables from 'env' file
load_dotenv(dotenv_path="env")

# Get Zyla API key from environment variables
zyla_api_key = os.getenv("ZYLA_GOOGLE_LENS_API_KEY")
if not zyla_api_key:
    print(
        "Warning: ZYLA_GOOGLE_LENS_API_KEY not found in env file. /search-lens endpoint will likely fail."
    )

# Get Public Base URL for constructing accessible image URLs
public_base_url = os.getenv("PUBLIC_BASE_URL")
if not public_base_url:
    print(
        "Warning: PUBLIC_BASE_URL not found in env file. /search-lens endpoint will fail if requires external access."
    )

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    # Import necessary components directly
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.dwpose import DWposeDetector
    from src.pose_guider import PoseGuider
    from src.utils_mask import get_mask_location
    from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
    from src.transformer_sd3_garm import (
        SD3Transformer2DModel as SD3Transformer2DModel_Garm,
    )
    from src.transformer_sd3_vton import (
        SD3Transformer2DModel as SD3Transformer2DModel_Vton,
    )

    # --- Import Gemini Functions ---
    try:
        from gemini_vision import (
            initialize_gemini_client,
            call_gemini_api,
            parse_gemini_response,
        )

        gemini_available = True
        print("Successfully imported Gemini vision functions.")
    except ImportError as gemini_e:
        print(f"Warning: Failed to import Gemini vision functions: {gemini_e}")
        print("The /generate-gemini endpoint will not be available.")
        initialize_gemini_client = None
        call_gemini_api = None
        parse_gemini_response = None
        gemini_available = False
    # --- End Import Gemini Functions ---

except ImportError as e:
    print(f"Error importing FitDiT components: {e}")
    # Set components to None so initialization check fails gracefully
    Parsing = None
    DWposeDetector = None
    PoseGuider = None
    get_mask_location = None
    StableDiffusion3TryOnPipeline = None
    SD3Transformer2DModel_Garm = None
    SD3Transformer2DModel_Vton = None
    CLIPVisionModelWithProjection = None
    gemini_available = False


# Define static folder path and ensure it exists
STATIC_FOLDER = os.path.join(current_dir, "static")
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)

# --- Global Storage for Preprocessed Data ---
preprocessed_data = {
    "vton_img": None,  # PIL Image
    "Upper-body": {"mask": None, "pose": None},
    "Lower-body": {"mask": None, "pose": None},
    "Dresses": {"mask": None, "pose": None},
    "lock": threading.Lock(),
}

# --- Helper Image Functions (copied from gradio_sd3.py) ---


def resize_image(img, target_size=768):
    width, height = img.size

    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Use PIL.Image.Resampling.LANCZOS for newer Pillow versions
    resampling_filter = (
        Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS
    )
    resized_img = img.resize((new_width, new_height), resampling_filter)

    return resized_img


def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255)):
    old_width, old_height = im.size

    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)

    # Use PIL.Image.Resampling.LANCZOS for newer Pillow versions
    resampling_filter = (
        Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS
    )
    im_resized = im.resize(new_size, resampling_filter)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    # Handle different image modes for padding color
    if im.mode == "RGBA":
        pad_color_with_alpha = pad_color + (255,) if len(pad_color) == 3 else pad_color
        new_im = Image.new("RGBA", (new_width, new_height), pad_color_with_alpha)
    elif im.mode == "L":
        # For grayscale mask, pad with black (0) or white (255) depending on context
        # Assuming black padding (0) for mask as in original process function
        new_im = Image.new("L", (new_width, new_height), 0)
    else:  # Assume RGB
        new_im = Image.new("RGB", (new_width, new_height), pad_color)

    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h


def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size

    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h

    # Ensure crop box is valid
    if left >= right or top >= bottom:
        print(
            f"Warning: Invalid crop box ({left}, {top}, {right}, {bottom}). Returning original padded image."
        )
        # Optionally return the original size blank image or the padded image itself
        return padded_im.resize(
            (original_width, original_height), Image.Resampling.LANCZOS
        )

    cropped_im = padded_im.crop((left, top, right, bottom))

    # Use PIL.Image.Resampling.LANCZOS for newer Pillow versions
    resampling_filter = (
        Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS
    )
    resized_im = cropped_im.resize((original_width, original_height), resampling_filter)

    return resized_im


# --- FitDiTGenerator Class ---
class FitDiTGenerator:
    def __init__(
        self,
        model_root,
        offload=False,
        aggressive_offload=False,
        device="cuda:0",
        with_fp16=False,
    ):
        # Check if necessary components were imported successfully
        if not all(
            [
                Parsing,
                DWposeDetector,
                PoseGuider,
                get_mask_location,
                StableDiffusion3TryOnPipeline,
                SD3Transformer2DModel_Garm,
                SD3Transformer2DModel_Vton,
                CLIPVisionModelWithProjection,
            ]
        ):
            raise ImportError(
                "One or more required FitDiT components failed to import. Cannot initialize generator."
            )

        weight_dtype = torch.float16 if with_fp16 else torch.bfloat16
        self.device = device
        self.weight_dtype = weight_dtype

        print("Initializing FitDiT components...")
        init_comp_start = time.time()
        # Load models (consider adding error handling for each step)
        try:
            print("  Loading transformer_garm...")
            self.transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
                os.path.join(model_root, "transformer_garm"), torch_dtype=weight_dtype
            )
            print("  Loading transformer_vton...")
            self.transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
                os.path.join(model_root, "transformer_vton"), torch_dtype=weight_dtype
            )
            print("  Loading pose_guider...")
            self.pose_guider = PoseGuider(
                conditioning_embedding_channels=1536,
                conditioning_channels=3,
                block_out_channels=(32, 64, 256, 512),
            )
            self.pose_guider.load_state_dict(
                torch.load(
                    os.path.join(
                        model_root, "pose_guider", "diffusion_pytorch_model.bin"
                    ),
                    map_location="cpu",
                )
            )  # Load to CPU first
            print("  Loading image_encoder_large (CLIP)...")
            self.image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", torch_dtype=weight_dtype
            )
            print("  Loading image_encoder_bigG (CLIP)...")
            self.image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype
            )

            # Move components to target device (after potential offloading setup)
            self.pose_guider.to(device=self.device, dtype=self.weight_dtype)
            self.image_encoder_large.to(device=self.device)
            self.image_encoder_bigG.to(device=self.device)

            print("  Initializing StableDiffusion3TryOnPipeline...")
            self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
                model_root,
                torch_dtype=weight_dtype,
                transformer_garm=self.transformer_garm,
                transformer_vton=self.transformer_vton,
                pose_guider=self.pose_guider,
                image_encoder_large=self.image_encoder_large,
                image_encoder_bigG=self.image_encoder_bigG,
            )
            print("  Pipeline initialized.")
        except Exception as e:
            print(f"Error loading a model component: {e}")
            print(traceback.format_exc())
            raise RuntimeError(
                f"Failed to load model components from {model_root}"
            ) from e

        init_comp_end = time.time()
        print(
            f"  Component loading time: {init_comp_end - init_comp_start:.2f} seconds"
        )

        # Preprocessors (load to CPU if offloading, otherwise target device)
        preprocessor_device = "cpu" if offload or aggressive_offload else self.device
        print(f"  Loading DWposeDetector to {preprocessor_device}...")
        self.dwprocessor = DWposeDetector(
            model_root=model_root, device=preprocessor_device
        )
        print(f"  Loading Parsing model to {preprocessor_device}...")
        self.parsing_model = Parsing(model_root=model_root, device=preprocessor_device)
        print("Preprocessors loaded.")

        # --- Load Segmentation Models ---
        segmentation_model_device = self.device
        print(f"  Loading YOLO model to {segmentation_model_device}...")
        try:
            # Using yolov8n (nano) for potentially faster inference
            self.yolo_model = YOLO("yolov8n.pt")
            # Move model to device if applicable (YOLO handles device internally during call)
            # self.yolo_model.to(segmentation_model_device) # YOLO handles this
            print("  YOLO model loaded.")
        except Exception as e:
            print(
                f"Error loading YOLO model: {e}. Original /generate endpoint segmentation might fail."
            )
            self.yolo_model = None

        print(f"  Loading FastSAM model to {segmentation_model_device}...")
        try:
            # Using FastSAM-x (large) - consider smaller if needed
            self.sam_model = FastSAM("FastSAM-x.pt")
            # Move model to device if applicable (FastSAM handles device internally during call)
            # self.sam_model.to(segmentation_model_device) # FastSAM handles this
            print("  FastSAM model loaded.")
        except Exception as e:
            print(
                f"Error loading FastSAM model: {e}. Segmentation will be skipped in both endpoints."
            )
            self.sam_model = None
        # --- End Load Segmentation Models ---

        # --- Initialize Gemini Client ---
        self.gemini_client = None
        self.gemini_model_name = GEMINI_MODEL_NAME
        if gemini_available and initialize_gemini_client:
            print("Initializing Gemini Client...")
            self.gemini_client = initialize_gemini_client()
            if self.gemini_client:
                print(
                    f"  Gemini client initialized for model {self.gemini_model_name}."
                )
            else:
                print(
                    "  Failed to initialize Gemini client. /generate-gemini endpoint will not work."
                )
        else:
            print(
                "Gemini functions not available, skipping Gemini client initialization."
            )
        # --- End Initialize Gemini Client ---

        # Setup offloading or move pipeline to device
        if offload:
            print("Enabling standard model CPU offload.")
            self.pipeline.enable_model_cpu_offload()
        elif aggressive_offload:
            print("Enabling aggressive sequential CPU offload.")
            self.pipeline.enable_sequential_cpu_offload()
        else:
            print(f"Moving pipeline components to {self.device}...")
            self.pipeline.to(self.device)

    # --- Internal Preprocessing Logic ---
    def _preprocess_images(
        self,
        vton_img_pil,
        category,
        offset_top,
        offset_bottom,
        offset_left,
        offset_right,
    ):
        if not self.dwprocessor or not self.parsing_model or not get_mask_location:
            raise RuntimeError(
                "Preprocessing components (DWPose, Parsing, get_mask_location) not available."
            )

        with torch.inference_mode():
            # 1. Resize for detection
            vton_img_det = resize_image(vton_img_pil)  # Default target_size=768

            # 2. Pose estimation
            pose_start = time.time()
            # DWProcessor expects BGR numpy array
            vton_img_det_np_bgr = np.array(vton_img_det)[:, :, ::-1]
            pose_image_np, _, _, candidate = self.dwprocessor(vton_img_det_np_bgr)
            pose_end = time.time()
            print(
                f"  Time - Pose Estimation (dwprocessor): {pose_end - pose_start:.2f} seconds"
            )

            if candidate is not None and len(candidate) > 0:
                candidate = candidate[0]  # Assuming single person
                candidate[candidate < 0] = 0
                # Scale keypoints to detection image size
                candidate[:, 0] = np.clip(
                    candidate[:, 0] * vton_img_det.width, 0, vton_img_det.width - 1
                )
                candidate[:, 1] = np.clip(
                    candidate[:, 1] * vton_img_det.height, 0, vton_img_det.height - 1
                )
            else:
                print(
                    "Warning: No pose detected by dwprocessor. Using default zero keypoints."
                )
                # Provide zero keypoints if none detected, crucial for get_mask_location
                candidate = np.zeros((18, 3))  # Assuming 18 keypoints structure

            # Pose image is RGB numpy array, convert to PIL
            pose_image_pil = Image.fromarray(
                pose_image_np[:, :, ::-1]
            )  # Convert BGR->RGB

            # 3. Human parsing
            parse_start = time.time()
            model_parse, _ = self.parsing_model(vton_img_det)
            parse_end = time.time()
            print(
                f"  Time - Human Parsing (parsing_model): {parse_end - parse_start:.2f} seconds"
            )

            # 4. Mask calculation using keypoints from detection size
            mask_loc_start = time.time()
            mask_pil, mask_gray_pil = get_mask_location(
                category,
                model_parse,  # The segmentation map (PIL Image)
                candidate,  # Keypoints scaled to model_parse dimensions
                model_parse.width,
                model_parse.height,
                offset_top,
                offset_bottom,
                offset_left,
                offset_right,
            )
            mask_loc_end = time.time()
            print(
                f"  Time - Mask Location Calculation (get_mask_location): {mask_loc_end - mask_loc_start:.2f} seconds"
            )

            # 5. Resize mask to original vton image size
            mask_pil = mask_pil.resize(
                vton_img_pil.size, Image.Resampling.NEAREST
            ).convert("L")

            # Return the final mask and pose image (as PIL images)
            return mask_pil, pose_image_pil

    # --- Internal Inference Logic ---
    def _run_inference(
        self,
        vton_img_pil,
        garm_img_pil,
        mask_pil,
        pose_image_pil,
        n_steps,
        image_scale,
        seed,
        num_images_per_prompt,
        resolution,
    ):
        if not self.pipeline:
            raise RuntimeError("Inference pipeline not available.")

        assert resolution in [
            "768x1024",
            "1152x1536",
            "1536x2048",
        ], "Invalid resolution."
        new_width, new_height = map(int, resolution.split("x"))

        with torch.inference_mode():
            # 1. Pre-inference Resizing/Padding
            pre_inf_start = time.time()
            original_size = vton_img_pil.size  # Store original size for unpadding

            # Pad/Resize inputs to the target inference resolution
            garm_img_resized, _, _ = pad_and_resize(
                garm_img_pil, new_width=new_width, new_height=new_height
            )
            vton_img_resized, pad_w, pad_h = pad_and_resize(
                vton_img_pil, new_width=new_width, new_height=new_height
            )
            # Pad mask (ensure it's L mode before padding)
            mask_resized, _, _ = pad_and_resize(
                mask_pil.convert("L"),  # Ensure L mode
                new_width=new_width,
                new_height=new_height,
                # pad_color should be 0 (black) for mask based on original logic
                # pad_and_resize handles 'L' mode now assuming 0 padding
            )
            mask_resized = mask_resized.convert("L")  # Ensure L mode after padding
            # Pad pose image
            pose_image_resized, _, _ = pad_and_resize(
                pose_image_pil,
                new_width=new_width,
                new_height=new_height,
                pad_color=(0, 0, 0),  # Black padding for pose
            )
            pre_inf_end = time.time()
            print(
                f"  Time - Pre-inference Resizing/Padding: {pre_inf_end - pre_inf_start:.2f} seconds"
            )

            # 2. Setup Seed and Generator
            current_seed = seed if seed != -1 else random.randint(0, 2147483647)
            print(f"Using seed: {current_seed}")
            # Use device='cpu' for generator for reproducibility across devices potentially
            generator_torch = torch.Generator(device="cpu").manual_seed(current_seed)

            # 3. Actual pipeline call
            pipeline_start = time.time()
            gen_images = self.pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=generator_torch,
                cloth_image=garm_img_resized,
                model_image=vton_img_resized,
                mask=mask_resized,  # Pass the L mode mask
                pose_image=pose_image_resized,
                num_images_per_prompt=num_images_per_prompt,
            ).images
            pipeline_end = time.time()
            print(
                f"  Time - Core Pipeline Inference: {pipeline_end - pipeline_start:.2f} seconds"
            )

            if not gen_images:
                print("Error: Model generated no images.")
                return None  # Indicate failure

            # 4. Post-processing (Unpad, Resize)
            post_proc_start = time.time()
            # Process only the first generated image if multiple were requested
            result_img_pil = unpad_and_resize(
                gen_images[0], pad_w, pad_h, original_size[0], original_size[1]
            )
            post_proc_end = time.time()
            print(
                f"  Time - Post-inference Unpadding/Resizing: {post_proc_end - post_proc_start:.2f} seconds"
            )

            return result_img_pil

    # --- Original Segmentation Method (using YOLO + FastSAM) ---
    def _segment_person(self, vton_img_pil: Image.Image) -> Image.Image:
        """Segments the largest person using YOLO detection + FastSAM segmentation."""
        if not self.yolo_model or not self.sam_model:
            print("Warning: YOLO or FastSAM model not loaded. Skipping segmentation.")
            return vton_img_pil

        print("Starting person segmentation (YOLO + FastSAM)...")
        seg_start_time = time.time()

        try:
            # 1. Convert PIL (RGB) to OpenCV (BGR)
            vton_img_cv = cv2.cvtColor(np.array(vton_img_pil), cv2.COLOR_RGB2BGR)
            h_orig, w_orig = vton_img_cv.shape[:2]

            # 2. Run YOLO detection
            print("  Running YOLO detection...")
            yolo_start = time.time()
            yolo_results = self.yolo_model(
                vton_img_cv, device=self.device, verbose=False
            )
            yolo_end = time.time()
            print(f"    YOLO detection time: {yolo_end - yolo_start:.2f}s")

            # 3. Find the largest 'person' bounding box
            largest_person_box = None
            max_area = 0
            if yolo_results and len(yolo_results) > 0:
                boxes = yolo_results[0].boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Class 0 is 'person'
                        xyxy_box = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy_box)
                        if x2 > x1 and y2 > y1:
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
                                largest_person_box = [x1, y1, x2, y2]

            if not largest_person_box:
                print("  No persons detected by YOLO. Skipping segmentation.")
                return vton_img_pil

            print(f"  Found largest person box (YOLO): {largest_person_box}")

            # 4. Run FastSAM segmentation using BBox Prompt
            print("  Running FastSAM segmentation (prompted by YOLO box)...")
            sam_start = time.time()
            sam_results = self.sam_model(
                vton_img_cv,
                device=self.device,
                retina_masks=True,
                conf=0.4,
                iou=0.9,
                bboxes=[largest_person_box],
                verbose=False,
            )
            sam_end = time.time()
            print(f"    FastSAM segmentation time: {sam_end - sam_start:.2f}s")

            # 5. Process FastSAM results and apply mask
            if (
                sam_results
                and len(sam_results) > 0
                and sam_results[0].masks is not None
            ):
                print("  FastSAM processing successful. Applying mask...")
                mask_data = sam_results[0].masks.data[0]
                mask_np = mask_data.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(
                    mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized.astype(bool)
                white_bg = np.ones_like(vton_img_cv, dtype=np.uint8) * 255
                segmented_img_cv = np.where(mask_bool[..., None], vton_img_cv, white_bg)
                segmented_img_pil = Image.fromarray(
                    cv2.cvtColor(segmented_img_cv, cv2.COLOR_BGR2RGB)
                )
                seg_end_time = time.time()
                print(
                    f"Segmentation (YOLO+FastSAM) finished in {seg_end_time - seg_start_time:.2f} seconds."
                )
                return segmented_img_pil
            else:
                print(
                    "  FastSAM did not produce results for the prompted bounding box (YOLO). Skipping segmentation."
                )
                return vton_img_pil

        except Exception as e:
            print(f"Error during segmentation (YOLO+FastSAM): {traceback.format_exc()}")
            print("  Segmentation failed. Returning original image.")
            return vton_img_pil

    # --- End Original Segmentation Method ---

    # --- New Segmentation Method (using Gemini + FastSAM) ---
    def _segment_person_gemini(self, vton_img_pil: Image.Image) -> Image.Image:
        """Segments the largest person using Gemini detection + FastSAM segmentation."""
        if not self.gemini_client or not self.sam_model:
            print(
                "Warning: Gemini client or FastSAM model not available. Skipping Gemini segmentation."
            )
            return vton_img_pil
        if not call_gemini_api or not parse_gemini_response:
            print(
                "Warning: Gemini API/parsing functions not available. Skipping Gemini segmentation."
            )
            return vton_img_pil

        print("Starting person segmentation (Gemini + FastSAM)...")
        seg_start_time = time.time()

        try:
            # 1. Call Gemini API to detect persons
            print("  Calling Gemini API for person detection...")
            gemini_start = time.time()
            width, height = vton_img_pil.size
            try:
                response_text = call_gemini_api(
                    self.gemini_client, self.gemini_model_name, "person", vton_img_pil
                )
            except Exception as api_err:
                print(f"  Gemini API call failed: {api_err}")
                return vton_img_pil  # Return original on API error

            gemini_end = time.time()
            print(f"    Gemini API call time: {gemini_end - gemini_start:.2f}s")

            # 2. Parse Gemini Response
            print("  Parsing Gemini response...")
            parse_start = time.time()
            bounding_boxes = parse_gemini_response(response_text, width, height)
            parse_end = time.time()
            print(f"    Gemini parsing time: {parse_end - parse_start:.2f}s")

            if not bounding_boxes:
                print("  No persons detected by Gemini. Skipping segmentation.")
                return vton_img_pil

            # 3. Find the largest 'person' bounding box from Gemini results
            largest_person_box = None
            max_area = 0
            for box_info in bounding_boxes:
                # Expecting box_2d = [x1, y1, x2, y2] in pixel coords
                x1, y1, x2, y2 = box_info["box_2d"]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_person_box = [x1, y1, x2, y2]  # Keep pixel coords

            if not largest_person_box:
                # This case should ideally not happen if bounding_boxes is not empty
                print(
                    "  Could not determine largest person box from Gemini results. Skipping segmentation."
                )
                return vton_img_pil

            print(f"  Found largest person box (Gemini): {largest_person_box}")
            # Optional: Draw Gemini box for debugging
            # try:
            #     debug_img = vton_img_pil.copy()
            #     draw = ImageDraw.Draw(debug_img)
            #     draw.rectangle(largest_person_box, outline="red", width=3)
            #     debug_img.save("gemini_detected_box.png")
            # except Exception as draw_err:
            #     print(f"Could not draw debug box: {draw_err}")

            # 4. Run FastSAM segmentation using the Gemini BBox Prompt
            print("  Running FastSAM segmentation (prompted by Gemini box)...")
            sam_start = time.time()
            vton_img_cv = cv2.cvtColor(np.array(vton_img_pil), cv2.COLOR_RGB2BGR)
            h_orig, w_orig = vton_img_cv.shape[:2]  # Get dimensions again just in case

            sam_results = self.sam_model(
                vton_img_cv,  # Use the OpenCV version for SAM
                device=self.device,
                retina_masks=True,
                conf=0.4,
                iou=0.9,
                bboxes=[largest_person_box],  # Use the box found by Gemini
                verbose=False,
            )
            sam_end = time.time()
            print(f"    FastSAM segmentation time: {sam_end - sam_start:.2f}s")

            # 5. Process FastSAM results and apply mask (same logic as before)
            if (
                sam_results
                and len(sam_results) > 0
                and sam_results[0].masks is not None
            ):
                print("  FastSAM processing successful. Applying mask...")
                mask_data = sam_results[0].masks.data[0]
                mask_np = mask_data.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(
                    mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized.astype(bool)
                white_bg = np.ones_like(vton_img_cv, dtype=np.uint8) * 255
                segmented_img_cv = np.where(mask_bool[..., None], vton_img_cv, white_bg)
                segmented_img_pil = Image.fromarray(
                    cv2.cvtColor(segmented_img_cv, cv2.COLOR_BGR2RGB)
                )
                seg_end_time = time.time()
                print(
                    f"Segmentation (Gemini+FastSAM) finished in {seg_end_time - seg_start_time:.2f} seconds."
                )
                return segmented_img_pil
            else:
                print(
                    "  FastSAM did not produce results for the prompted bounding box (Gemini). Skipping segmentation."
                )
                return vton_img_pil

        except Exception as e:
            print(
                f"Error during segmentation (Gemini+FastSAM): {traceback.format_exc()}"
            )
            print("  Segmentation failed. Returning original image.")
            return vton_img_pil

    # --- End New Segmentation Method ---


# --- Instantiate the generator ---
generator = None
# Ensure all necessary components were imported before trying to instantiate
if all(
    [
        Parsing,
        DWposeDetector,
        PoseGuider,
        get_mask_location,
        StableDiffusion3TryOnPipeline,
        SD3Transformer2DModel_Garm,
        SD3Transformer2DModel_Vton,
        CLIPVisionModelWithProjection,
    ]
):
    try:
        print(f"Initializing FitDiTGenerator with model path: {MODEL_PATH}")
        init_start_time = time.time()
        generator = FitDiTGenerator(
            MODEL_PATH, OFFLOAD, AGGRESSIVE_OFFLOAD, DEVICE, WITH_FP16
        )
        init_end_time = time.time()
        print(
            f"FitDiTGenerator initialized successfully in {init_end_time - init_start_time:.2f} seconds."
        )
    except Exception as e:
        print(f"Error initializing FitDiTGenerator: {e}")
        print(f"Detailed error: {traceback.format_exc()}")
        generator = None
else:
    print(
        "One or more required FitDiT components could not be imported. Server cannot initialize model."
    )


# --- Helper to encode PIL image to base64 string ---
def encode_pil_to_base64(pil_image, format="PNG"):
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# --- Helper to decode base64 string to PIL image ---
def decode_base64_to_pil(base64_string):
    img_data = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(img_data))
    return pil_image


# --- Zyla Google Lens API Function (Integrated) ---
def search_google_lens(image_url):
    """
    Search for an image using Google Lens API via Zyla Labs

    Args:
        image_url (str): URL of the image to search

    Returns:
        dict: JSON response from the API or None on error
    """
    global zyla_api_key  # Access the globally loaded key
    if not zyla_api_key:
        print("Error: Zyla API key is not configured.")
        return None

    # API endpoint
    endpoint = "https://zylalabs.com/api/1338/google+lens+search+api/1119/search"

    # Request headers
    headers = {
        "Authorization": f"Bearer {zyla_api_key}",
        "Content-Type": "application/json",
    }

    # Request parameters
    params = {"url": image_url, "country": "us"}

    try:
        # Make the request
        response = requests.get(
            endpoint, headers=headers, params=params, timeout=30
        )  # Added timeout

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Zyla API Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during Zyla API request: {e}")
        return None


# --- API Endpoints ---


@app.route("/preprocess", methods=["POST"])
def preprocess_images():
    request_start_time = time.time()
    print("--- New /preprocess Request (Batched Categories) ---")

    if generator is None:
        return jsonify({"error": "Model generator not initialized."}), 503

    # --- Input Reading (JSON with base64 vton_img) ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "vton_img_base64" not in data:
        return jsonify({"error": "Missing 'vton_img_base64' in JSON payload"}), 400

    try:
        # Decode vton image
        vton_img_pil = decode_base64_to_pil(data["vton_img_base64"]).convert("RGB")
        # Read optional offsets (can still be useful)
        offset_top = int(data.get("offset_top", 0))
        offset_bottom = int(data.get("offset_bottom", 0))
        offset_left = int(data.get("offset_left", 0))
        offset_right = int(data.get("offset_right", 0))

    except Exception as e:
        print(f"Error processing input or decoding base64: {traceback.format_exc()}")
        return (
            jsonify({"error": f"Bad request data or invalid base64 string: {e}"}),
            400,
        )

    # --- Preprocessing Step (Parallel for 3 categories) ---
    preprocess_start_time = time.time()
    categories = ["Upper-body", "Lower-body", "Dresses"]
    results = {}

    # Helper function to run preprocessing for one category
    def run_preprocess_for_category(category):
        try:
            print(f"  Starting preprocessing for category: {category}")
            cat_start_time = time.time()
            mask_pil, pose_image_pil = generator._preprocess_images(
                vton_img_pil,
                category,
                offset_top,
                offset_bottom,
                offset_left,
                offset_right,
            )
            cat_end_time = time.time()
            print(
                f"  Finished preprocessing for category: {category} in {cat_end_time - cat_start_time:.2f}s"
            )
            return category, mask_pil, pose_image_pil
        except Exception as e:
            print(f"Error preprocessing category {category}: {traceback.format_exc()}")
            return category, None, None  # Return None on failure for this category

    # Use ThreadPoolExecutor for potential concurrency
    # Max workers = 3 since we have 3 categories
    all_successful = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_category = {
            executor.submit(run_preprocess_for_category, cat): cat for cat in categories
        }
        for future in concurrent.futures.as_completed(future_to_category):
            category = future_to_category[future]
            try:
                cat, mask, pose = future.result()
                if mask is None or pose is None:
                    all_successful = False
                    results[cat] = {
                        "mask": None,
                        "pose": None,
                    }  # Store failure indicator
                else:
                    results[cat] = {"mask": mask, "pose": pose}
            except Exception as exc:
                print(f"Category {category} generated an exception: {exc}")
                all_successful = False
                results[cat] = {"mask": None, "pose": None}  # Store failure indicator

    preprocess_end_time = time.time()
    print(
        f"Time - Preprocessing Step Total (All Categories): {preprocess_end_time - preprocess_start_time:.2f} seconds"
    )

    # --- Update Global Storage ---
    update_start = time.time()
    with preprocessed_data["lock"]:
        preprocessed_data["vton_img"] = vton_img_pil  # Store the new vton image
        for cat in categories:
            if cat in results:  # Store result (success or failure)
                preprocessed_data[cat] = results[cat]
            else:  # Should not happen with as_completed, but safety check
                preprocessed_data[cat] = {"mask": None, "pose": None}
    update_end = time.time()
    print(f"Time - Updating Global Storage: {update_end - update_start:.2f} seconds")

    request_end_time = time.time()
    print(
        f"Time - Total /preprocess Request: {request_end_time - request_start_time:.2f} seconds"
    )

    if not all_successful:
        print("--- /preprocess Request Completed with Errors for some categories ---")
        # Still return 200, but client might want to check /infer response later
        return jsonify({"status": "preprocessing_completed_with_errors"})
    else:
        print("--- /preprocess Request Completed Successfully ---")
        return jsonify({"status": "preprocessing_complete"})


@app.route("/infer", methods=["POST"])
def infer_tryon():
    request_start_time = time.time()
    print("--- New /infer Request (Using Stored Preprocessed Data) ---")

    if generator is None:
        return jsonify({"error": "Model generator not initialized."}), 503

    # --- Input Reading (JSON: garm_img_base64, category, options) ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_fields = ["garm_img_base64", "category"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing '{field}' in JSON payload"}), 400

    category = data["category"]
    if category not in ["Upper-body", "Lower-body", "Dresses"]:
        return jsonify({"error": "Invalid category."}), 400

    try:
        # Read parameters from JSON data
        n_steps = int(data.get("n_steps", 20))
        image_scale = float(data.get("image_scale", 2.0))
        seed = int(data.get("seed", -1))
        num_images_per_prompt = 1  # Keep fixed for this endpoint
        resolution = data.get("resolution", "768x1024")
        if resolution not in ["768x1024", "1152x1536", "1536x2048"]:
            return jsonify({"error": "Invalid resolution."}), 400

        # Decode garment image from base64 string
        garm_img_pil = decode_base64_to_pil(data["garm_img_base64"]).convert("RGB")

    except Exception as e:
        print(f"Error processing input or decoding base64: {traceback.format_exc()}")
        return (
            jsonify({"error": f"Bad request data or invalid base64 string: {e}"}),
            400,
        )

    # --- Retrieve Preprocessed Data from Global Storage ---
    retrieval_start = time.time()
    with preprocessed_data["lock"]:
        vton_img_pil = preprocessed_data.get("vton_img")
        category_data = preprocessed_data.get(category)
        mask_pil = category_data.get("mask") if category_data else None
        pose_image_pil = category_data.get("pose") if category_data else None
    retrieval_end = time.time()
    print(
        f"Time - Retrieving data from global storage: {retrieval_end - retrieval_start:.2f} seconds"
    )

    if vton_img_pil is None:
        return (
            jsonify({"error": "VTON image not found. Please run /preprocess first."}),
            400,
        )
    if mask_pil is None or pose_image_pil is None:
        return (
            jsonify(
                {
                    "error": f"Preprocessed mask/pose for category '{category}' not found or preprocessing failed. Please run /preprocess."
                }
            ),
            400,
        )

    print(
        f"Processing /infer request: category={category}, resolution={resolution}, seed={seed}"
    )

    # --- Inference Step ---
    inference_start_time = time.time()
    try:
        result_img_pil = generator._run_inference(
            vton_img_pil,  # From global storage
            garm_img_pil,  # From request
            mask_pil,  # From global storage
            pose_image_pil,  # From global storage
            n_steps,
            image_scale,
            seed,
            num_images_per_prompt,
            resolution,
        )
        inference_end_time = time.time()
        print(
            f"Time - Inference Step Total: {inference_end_time - inference_start_time:.2f} seconds"
        )

        if result_img_pil is None:
            return jsonify({"error": "Model failed to generate an image."}), 500

        # --- Encode result to base64 and send JSON response ---
        encode_start = time.time()
        result_base64 = encode_pil_to_base64(result_img_pil, format="PNG")
        encode_end = time.time()
        print(
            f"Time - Encoding Result to Base64: {encode_end - encode_start:.2f} seconds"
        )

        request_end_time = time.time()
        print(
            f"Time - Total /infer Request: {request_end_time - request_start_time:.2f} seconds"
        )
        print(f"--- /infer Request Completed ---")
        return jsonify({"result_image_base64": result_base64})

    except Exception as e:
        print(f"Error during inference: {traceback.format_exc()}")
        request_end_time = time.time()
        print(
            f"Time - Total /infer Request (Error): {request_end_time - request_start_time:.2f} seconds"
        )
        print("--- /infer Request Failed ---")
        return jsonify({"error": f"Internal server error during inference: {e}"}), 500


@app.route("/generate", methods=["POST"])
def generate_tryon():
    request_start_time = time.time()
    print("--- New /generate Request (Combined Preprocess & Infer - YOLO/SAM) ---")

    if generator is None:
        return jsonify({"error": "Model generator not initialized."}), 503

    # --- Input Reading (JSON: vton_img, garm_img, category, options) ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_fields = ["vton_img_base64", "garm_img_base64", "category"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing '{field}' in JSON payload"}), 400

    category = data["category"]
    if category not in ["Upper-body", "Lower-body", "Dresses"]:
        return jsonify({"error": "Invalid category."}), 400

    try:
        # Decode images
        vton_img_pil_orig = decode_base64_to_pil(data["vton_img_base64"]).convert("RGB")
        garm_img_pil = decode_base64_to_pil(data["garm_img_base64"]).convert("RGB")

        # --- Use Original Segmentation Step (YOLO + FastSAM) ---
        vton_img_pil = generator._segment_person(vton_img_pil_orig)
        # --- End Segmentation Step ---

        # +++ Encode original VTON image +++
        original_vton_base64 = encode_pil_to_base64(vton_img_pil_orig, format="PNG")
        # +++ End Encode original VTON image +++

        # Read optional preprocessing offsets (though less critical here)
        offset_top = int(data.get("offset_top", 0))
        offset_bottom = int(data.get("offset_bottom", 0))
        offset_left = int(data.get("offset_left", 0))
        offset_right = int(data.get("offset_right", 0))

        # Read inference parameters
        n_steps = int(data.get("n_steps", 20))
        image_scale = float(data.get("image_scale", 2.0))
        seed = int(data.get("seed", -1))
        num_images_per_prompt = 1  # Keep fixed for simplicity
        resolution = data.get("resolution", "768x1024")
        if resolution not in ["768x1024", "1152x1536", "1536x2048"]:
            return jsonify({"error": "Invalid resolution."}), 400

    except Exception as e:
        print(f"Error processing input or decoding base64: {traceback.format_exc()}")
        return (
            jsonify({"error": f"Bad request data or invalid base64 string: {e}"}),
            400,
        )

    print(
        f"Processing /generate request: category={category}, resolution={resolution}, seed={seed}"
    )

    # --- Step 1: Preprocessing ---
    preprocess_start_time = time.time()
    try:
        print(f"  Starting preprocessing for category: {category}")
        mask_pil, pose_image_pil = generator._preprocess_images(
            vton_img_pil,
            category,
            offset_top,
            offset_bottom,
            offset_left,
            offset_right,
        )
        preprocess_end_time = time.time()
        if mask_pil is None or pose_image_pil is None:
            print(f"  Preprocessing failed for category: {category}")
            return (
                jsonify({"error": f"Preprocessing failed for category '{category}'."}),
                500,
            )
        print(
            f"  Finished preprocessing in {preprocess_end_time - preprocess_start_time:.2f}s"
        )
    except Exception as e:
        print(f"Error during preprocessing: {traceback.format_exc()}")
        return (
            jsonify({"error": f"Internal server error during preprocessing: {e}"}),
            500,
        )

    # --- Step 2: Inference ---
    inference_start_time = time.time()
    try:
        result_img_pil = generator._run_inference(
            vton_img_pil,
            garm_img_pil,
            mask_pil,
            pose_image_pil,
            n_steps,
            image_scale,
            seed,
            num_images_per_prompt,
            resolution,
        )
        inference_end_time = time.time()
        print(
            f"  Time - Inference Step Total: {inference_end_time - inference_start_time:.2f} seconds"
        )

        if result_img_pil is None:
            return jsonify({"error": "Model failed to generate an image."}), 500

        # --- Step 3: Encode result to base64 and send JSON response ---
        encode_start = time.time()
        masked_img_base64 = encode_pil_to_base64(result_img_pil, format="PNG")

        # --- Step 4: Composite result onto original background ---
        composite_start = time.time()
        overlaid_img_base64 = None
        try:
            result_np = np.array(result_img_pil.convert("RGB"))
            original_np = np.array(vton_img_pil_orig.convert("RGB"))

            if result_np.shape != original_np.shape:
                print(
                    f"Warning: Dimension mismatch for compositing (YOLO). Result: {result_np.shape}, Original: {original_np.shape}. Skipping."
                )
                original_img_base64 = None
            else:
                mask = np.any(result_np < 240, axis=-1)
                composite_np = np.where(mask[..., np.newaxis], result_np, original_np)
                composite_img_pil = Image.fromarray(composite_np.astype(np.uint8))
                overlaid_img_base64 = encode_pil_to_base64(
                    composite_img_pil, format="PNG"
                )

        except Exception as composite_exc:
            print(f"Error during compositing (YOLO): {traceback.format_exc()}")
            overlaid_img_base64 = None

        composite_end = time.time()
        print(
            f"  Time - Compositing onto Original (YOLO): {composite_end - composite_start:.2f} seconds"
        )
        encode_end = time.time()
        print(
            f"  Time - Encoding Results (YOLO): {encode_end - encode_start:.2f} seconds"
        )

        request_end_time = time.time()
        print(
            f"Time - Total /generate Request (YOLO): {request_end_time - request_start_time:.2f} seconds"
        )
        print("--- /generate Request (YOLO) Completed ---")
        response_data = {
            "masked_img_base64": masked_img_base64,
            "original_img_base64": original_vton_base64,
        }
        if overlaid_img_base64:
            response_data["overlaid_img_base64"] = overlaid_img_base64
        else:
            response_data["compositing_status"] = "failed_or_skipped"

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during inference (YOLO): {traceback.format_exc()}")
        request_end_time = time.time()
        print(
            f"Time - Total /generate Request (YOLO Error): {request_end_time - request_start_time:.2f} seconds"
        )
        print("--- /generate Request (YOLO) Failed ---")
        return jsonify({"error": f"Internal server error during inference: {e}"}), 500


# --- NEW /generate-gemini Endpoint (Gemini + FastSAM) ---
@app.route("/generate-gemini", methods=["POST"])
def generate_tryon_gemini():
    request_start_time = time.time()
    print(
        "--- New /generate-gemini Request (Combined Preprocess & Infer - Gemini/SAM) ---"
    )

    if generator is None:
        return jsonify({"error": "Model generator not initialized."}), 503
    if not generator.gemini_client:
        return (
            jsonify({"error": "Gemini client not available or not initialized."}),
            503,
        )

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ["vton_img_base64", "garm_img_base64", "category"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing '{field}' in JSON payload"}), 400

    category = data["category"]
    if category not in ["Upper-body", "Lower-body", "Dresses"]:
        return jsonify({"error": "Invalid category."}), 400

    try:
        # Decode images (same as original endpoint)
        vton_img_pil_orig = decode_base64_to_pil(data["vton_img_base64"]).convert("RGB")
        garm_img_pil = decode_base64_to_pil(data["garm_img_base64"]).convert("RGB")

        # --- Use NEW Segmentation Step (Gemini + FastSAM) ---
        vton_img_pil = generator._segment_person_gemini(vton_img_pil_orig)
        # --- End Segmentation Step ---

        # Encode original VTON image (same as original endpoint)
        original_vton_base64 = encode_pil_to_base64(vton_img_pil_orig, format="PNG")

        # Read optional preprocessing offsets (same as original endpoint)
        offset_top = int(data.get("offset_top", 0))
        offset_bottom = int(data.get("offset_bottom", 0))
        offset_left = int(data.get("offset_left", 0))
        offset_right = int(data.get("offset_right", 0))

        # Read inference parameters (same as original endpoint)
        n_steps = int(data.get("n_steps", 20))
        image_scale = float(data.get("image_scale", 2.0))
        seed = int(data.get("seed", -1))
        num_images_per_prompt = 1
        resolution = data.get("resolution", "768x1024")
        if resolution not in ["768x1024", "1152x1536", "1536x2048"]:
            return jsonify({"error": "Invalid resolution."}), 400

    except Exception as e:
        print(
            f"Error processing input or decoding base64 (Gemini): {traceback.format_exc()}"
        )
        return (
            jsonify({"error": f"Bad request data or invalid base64 string: {e}"}),
            400,
        )

    print(
        f"Processing /generate-gemini request: category={category}, resolution={resolution}, seed={seed}"
    )

    # --- Step 1: Preprocessing (same as original endpoint) ---
    # Note: Preprocessing uses the *segmented* vton_img_pil from above
    preprocess_start_time = time.time()
    try:
        print(f"  Starting preprocessing for category: {category} (Gemini flow)")
        mask_pil, pose_image_pil = generator._preprocess_images(
            vton_img_pil,  # Uses the image segmented by Gemini+SAM
            category,
            offset_top,
            offset_bottom,
            offset_left,
            offset_right,
        )
        preprocess_end_time = time.time()
        if mask_pil is None or pose_image_pil is None:
            print(f"  Preprocessing failed for category: {category} (Gemini flow)")
            return (
                jsonify({"error": f"Preprocessing failed for category '{category}'."}),
                500,
            )
        print(
            f"  Finished preprocessing (Gemini flow) in {preprocess_end_time - preprocess_start_time:.2f}s"
        )
    except Exception as e:
        print(f"Error during preprocessing (Gemini flow): {traceback.format_exc()}")
        return (
            jsonify({"error": f"Internal server error during preprocessing: {e}"}),
            500,
        )

    # --- Step 2: Inference (same as original endpoint) ---
    inference_start_time = time.time()
    try:
        result_img_pil = generator._run_inference(
            vton_img_pil,  # Uses the segmented image
            garm_img_pil,
            mask_pil,  # Generated from the segmented image
            pose_image_pil,  # Generated from the segmented image
            n_steps,
            image_scale,
            seed,
            num_images_per_prompt,
            resolution,
        )
        inference_end_time = time.time()
        print(
            f"  Time - Inference Step Total (Gemini flow): {inference_end_time - inference_start_time:.2f} seconds"
        )

        if result_img_pil is None:
            return jsonify({"error": "Model failed to generate an image."}), 500

        # --- Step 3: Encode result (same as original endpoint) ---
        encode_start = time.time()
        masked_img_base64 = encode_pil_to_base64(result_img_pil, format="PNG")

        # --- Step 4: Composite result onto original background (same as original endpoint) ---
        composite_start = time.time()
        overlaid_img_base64 = None
        try:
            result_np = np.array(result_img_pil.convert("RGB"))
            original_np = np.array(
                vton_img_pil_orig.convert("RGB")
            )  # Use the *original* unsegmented image

            if result_np.shape != original_np.shape:
                print(
                    f"Warning: Dimension mismatch for compositing (Gemini). Result: {result_np.shape}, Original: {original_np.shape}. Skipping."
                )
                # Indicate compositing was skipped or failed
                overlaid_img_base64 = None  # Keep it None
            else:
                # Mask becomes True for pixels that are NOT close to white (threshold 240)
                mask = np.any(result_np < 240, axis=-1)

                # Combine: where mask is True, use result; otherwise use original
                composite_np = np.where(mask[..., np.newaxis], result_np, original_np)

                # Convert back to PIL Image
                composite_img_pil = Image.fromarray(composite_np.astype(np.uint8))
                overlaid_img_base64 = encode_pil_to_base64(
                    composite_img_pil, format="PNG"
                )

        except Exception as composite_exc:
            print(f"Error during compositing (Gemini): {traceback.format_exc()}")
            overlaid_img_base64 = None  # Indicate failure

        composite_end = time.time()
        print(
            f"  Time - Compositing onto Original (Gemini): {composite_end - composite_start:.2f} seconds"
        )

        encode_end = time.time()
        print(
            f"  Time - Encoding Results (Gemini): {encode_end - encode_start:.2f} seconds"
        )

        request_end_time = time.time()
        print(
            f"Time - Total /generate-gemini Request: {request_end_time - request_start_time:.2f} seconds"
        )
        print("--- /generate-gemini Request Completed ---")
        # Return all three images (masked, original, overlaid)
        response_data = {
            "masked_img_base64": masked_img_base64,
            "original_img_base64": original_vton_base64,  # Add the original VTON
        }
        if overlaid_img_base64:
            response_data["overlaid_img_base64"] = overlaid_img_base64
        else:
            # Optionally signal issues
            response_data["compositing_status"] = "failed_or_skipped"

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during inference (Gemini flow): {traceback.format_exc()}")
        request_end_time = time.time()
        print(
            f"Time - Total /generate-gemini Request (Error): {request_end_time - request_start_time:.2f} seconds"
        )
        print("--- /generate-gemini Request Failed ---")
        return jsonify({"error": f"Internal server error during inference: {e}"}), 500


# --- /search-lens Endpoint --- (unchanged)
@app.route("/search-lens", methods=["POST"])
def search_lens_endpoint():
    print("--- New /search-lens Request ---")
    request_start_time = time.time()

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' in JSON payload"}), 400

    if not zyla_api_key:
        return (
            jsonify(
                {
                    "error": "Google Lens search function not available due to missing API key."
                }
            ),
            503,
        )

    if not public_base_url:
        return (
            jsonify({"error": "Server configuration error: PUBLIC_BASE_URL not set."}),
            500,
        )

    image_base64 = data["image_base64"]
    file_path = None

    try:
        try:
            if "," in image_base64:
                header, encoded = image_base64.split(",", 1)
            else:
                encoded = image_base64
            img_data = base64.b64decode(encoded)
            pil_image = Image.open(io.BytesIO(img_data))
            img_format = pil_image.format if pil_image.format else "PNG"
            img_format = img_format.lower()
            if img_format not in ["png", "jpeg", "jpg"]:
                print(
                    f"Warning: Unsupported image format '{img_format}', saving as PNG."
                )
                img_format = "png"

        except (base64.binascii.Error, ValueError, IOError) as e:
            print(f"Error decoding base64 or opening image: {e}")
            return jsonify({"error": "Invalid base64 string or image data"}), 400

        filename = f"{uuid.uuid4().hex}.{img_format}"
        file_path = os.path.join(STATIC_FOLDER, filename)
        try:
            pil_image.save(file_path)
            print(f"  Saved temporary image to: {file_path}")
        except Exception as e:
            print(f"Error saving image file: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to save temporary image: {e}"}), 500

        try:
            public_url = f"{public_base_url.rstrip('/')}/static/{filename}"
            print(f"  Generated public URL: {public_url}")
        except Exception as e:
            print(f"Error generating public URL: {traceback.format_exc()}")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_err:
                    print(
                        f"Error deleting temporary file during URL generation error: {remove_err}"
                    )
            return jsonify({"error": f"Failed to generate image URL: {e}"}), 500

        print(f"  Calling search_google_lens with URL: {public_url}")
        api_call_start = time.time()
        lens_result = search_google_lens(public_url)
        api_call_end = time.time()
        print(f"  Time - Zyla API Call: {api_call_end - api_call_start:.2f} seconds")

        request_end_time = time.time()
        print(
            f"Time - Total /search-lens Request: {request_end_time - request_start_time:.2f} seconds"
        )

        if lens_result:
            print("--- /search-lens Request Completed Successfully ---")
            return jsonify(lens_result)
        else:
            print("--- /search-lens Request Failed (API returned None) ---")
            return (
                jsonify(
                    {"error": "Google Lens API call failed or returned no result."}
                ),
                502,
            )

    except Exception as e:
        print(f"Unexpected error in /search-lens: {traceback.format_exc()}")
        request_end_time = time.time()
        print(
            f"Time - Total /search-lens Request (Error): {request_end_time - request_start_time:.2f} seconds"
        )
        return jsonify({"error": f"Internal server error: {e}"}), 500

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  Cleaned up temporary file: {file_path}")
            except OSError as e:
                print(f"Error deleting temporary file {file_path}: {e}")


if __name__ == "__main__":
    if generator is None:
        print("Failed to initialize model generator. Flask server cannot start.")
    else:
        print(f"Starting Flask server, listening on port 5000...")
        app.run(host="0.0.0.0", port=5000, debug=False)
