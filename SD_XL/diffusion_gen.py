import torch

import numpy as np

import cv2
from PIL import Image

# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Generate Library
import os
import requests
import json 
import requests
import base64
from io import BytesIO
from PIL import Image


def convert_to_base64(image):
  """
  Chuyển đổi ảnh sang dạng base64

  Args:
    image_path: Đường dẫn đến ảnh

  Returns:
    Chuỗi base64 của ảnh
  """

  # Đọc ảnh
  # image = cv2.imread(image_path)

  # Mã hóa ảnh sang dạng base64
  _, buffer = cv2.imencode('.jpg', image)
  base64_string = base64.b64encode(buffer.tobytes()).decode("utf-8")

  return base64_string


def convert_mask_to_base64(image):
  """
  Chuyển đổi ảnh đen trắng sang dạng base64

  Args:
    image_path: Đường dẫn đến ảnh

  Returns:
    Chuỗi base64 của ảnh
  """

  # Đọc ảnh
  # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  # Mã hóa ảnh sang dạng base64
  _, buffer = cv2.imencode('.jpg', image)
  base64_string = base64.b64encode(buffer.tobytes()).decode("utf-8")

  return base64_string


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

    def inpaintingApi(self, image, mask, width, height, prompt, negative_prompt, url, key):
        base64_img = convert_to_base64(image)
        base64_mask = convert_mask_to_base64(mask)
        payload = json.dumps({
            "key": key,
            "model_id":'epicrealism5',
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_image": str(base64_img),
            "mask_image": str(base64_mask),
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
            "base64": "yes",
            })
        
        headers = {
            'Content-Type': 'application/json'
            }

        response = requests.request("POST", url, headers=headers, data=payload)

        out = json.loads(response.text)
        print(out)
        
        return out
    
    def reloadImage(self, outSD, url_fetch, key): 
        payload = json.dumps({
        "key": key,
        "request_id": outSD['id']
        })

        headers = {
        'Content-Type': 'application/json'
        }

        response_out = requests.request("POST", url_fetch, headers=headers, data=payload)
        out_image = json.loads(response_out.text)
        print(out_image)
        return out_image
    
    def pipelineApi(self, image, mask, width, height, prompt, negative_prompt, url, url_fetch, key):
        outSD = self.inpaintingApi(image=image, mask=mask, width=width, height=height, prompt=prompt, negative_prompt=negative_prompt, url=url, key=key)
        status_outSD = None
        while status_outSD != 'success':
            outReload = self.reloadImage(outSD, url_fetch, key)
            status_outSD = outReload['status']
            print(status_outSD)
        output_url = str(outReload['output'][0])
        outImageUrl = output_url.replace('temp', 'generations')
        print(status_outSD)
        # if status_outSD == 'success': 
        #     output_url = str(outSD['output'][0])
        #     outImageUrl = output_url.replace('temp', 'generations')
        # else: 
        #     while status_outSD != 'success':
        #         outReload = self.reloadImage(outSD, url_fetch, key)
        #         status_outSD = outReload['status']
        #     output_url = str(outReload['output'][0])
        #     outImageUrl = output_url.replace('temp', 'generations')
        return outImageUrl
