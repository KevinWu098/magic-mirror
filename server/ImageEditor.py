#!/usr/bin/env python3
# pip install diffusers accelerate transformers xformers --upgrade

import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler
)
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32   = True

class ImageEditor:
    def __init__(self):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None
        )

        self.pipe.to("cuda")
        print("🚀 Image editor on device:", next(self.pipe.unet.parameters()).device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        random_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        # warm up the pipeline
        with torch.autocast("cuda"):
            self.pipe(
                prompt="hi",
                image=random_img,
                num_inference_steps=1,
                guidance_scale=9,
                image_guidance_scale=2,
                height=512,
                width=512
            )

    def get_edited_image(
        self,
        input_image: Image.Image,
        prompt: str = "Make this shirt red",
        height: int = 512,
        width: int = 512,
        steps: int = 20,
        prompt_guidance: float = 9,
        image_guidance: float = 2,
    ):  
        with torch.autocast("cuda"):
            result = self.pipe(
                prompt=prompt,
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=prompt_guidance,
                image_guidance_scale=image_guidance,
                height=height,
                width=width
            )
        return result.images[0]


if __name__ == "__main__":
    print("Creating image editor...")
    image_editor = ImageEditor()
    print("Editing image...")
    edited_image = image_editor.get_edited_image(Image.open("kev_globe_shirt.jpg"), "Make this shirt red", 512, 512, 20, 9, 2)
    print("Saving image...")
    edited_image.save("output.png")
    print("Done!")