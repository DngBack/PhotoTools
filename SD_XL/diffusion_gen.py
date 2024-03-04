import torch

import numpy as np

import cv2
from PIL import Image

# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

import requests
import json 

class DiffusionGenerationV2:
    """
    Stable Diffusion for generation process.
    Using Stable Diffusion 2.0 from stability

    """

    def __init__(self,  device):
        """
        Args:
            device (torch.device): Device used.
        """
        # Setup device
        self.device = device

    def load_module(self, module_path= "stabilityai/stable-diffusion-2-inpainting"):
        inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
            module_path,
            torch_dtype=torch.float32,
        )

        self.inpaint_pipe = inpaint_pipe.to(self.device)


    def inpaint_image(self, image, mask, prompt, negative_prompt):
        """
        Inpainting function

        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Mask image

        Returns:
            inpainted_image (PIL.Image): Inpainted image
        """
        # Save ori_size of input image to reconstruct
        ori_size = image.size

        # Resize image and mask to passing model
        input_image = image.resize((512, 512))
        input_mask = mask.resize((512, 512))

        # Apply pipeline
        generator = torch.Generator(self.device).manual_seed(42)
        result = self.inpaint_pipe(
            image=input_image,
            mask_image=input_mask,
            prompt= prompt,
            negative_prompt= negative_prompt,
            generator=generator,
        )
        output_image = result.images[0]

        # Resize inpainted image to original size
        inpainted_image = output_image.resize(ori_size)

        return inpainted_image


# Model Load with stablediffusionapi 
class DiffusionGenerationAPI:
    """
    Stable Diffusion for generation process.
    Using Stable Diffusion API 

    """

    def __init__(self,  device):
        """
        Args:
            device (torch.device): Device used.
        """
        # Setup device

        self.device = device

    def inpaint_image(self, image, mask, width, height, prompt, negative_prompt, url, url_fetch, key):
        payload = json.dumps({
            "key": key,
            "model_id":'epicrealism5',
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_image": image,
            "mask_image": mask,
            "width": str(width),
            "height": str(height),
            "samples": "1",
            "steps": "40",
            "safety_checker": "no",
            "enhance_prompt": "yes",
            "guidance_scale": 8.5,
            "strength": 1.0,
            "scheduler": "DPMSolverMultistepScheduler",
            "seed": None,
            "lora_model": 'more_details',
            "tomesd": "no",
            "use_karras_sigmas": "yes",
            "vae": 'sd-vae-ft-mse-original',
            "lora_strength": 0.5,
            "embeddings_model": None,
            "webhook": None,
            "track_id": None,
            "base": "yes",
            })
        
        headers = {
            'Content-Type': 'application/json'
            }

        response = requests.request("POST", url, headers=headers, data=payload)

        out = json.loads(response.text)

        payload = json.dumps({
        "key": key,
        "request_id": out['id']
        })

        headers = {
        'Content-Type': 'application/json'
        }

        response_out = requests.request("POST", url_fetch, headers=headers, data=payload)
        out_image = json.loads(response_out.text)
        outImageUrl = str(out_image[0]).replace('temp', 'generations')
        
        return outImageUrl