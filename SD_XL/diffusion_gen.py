import torch

import numpy as np

import cv2
from PIL import Image

# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
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
    

# Model with stablediffusionapi/epicdream
class DiffusionGenerationEpicDream:
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

    def load_module(self, module_path= "stablediffusionapi/epicdream"):
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stablediffusionapi/epicdream", torch_dtype=torch.float32
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
        width, height = image.size

        # Apply pipeline
        generator = torch.Generator(self.device).manual_seed(42)
        result = self.inpaint_pipe(
            image=image,
            mask_image=mask,
            prompt= prompt,
            negative_prompt= negative_prompt,
            height=height,
            width=width,
            generator=generator,
        )
        output_image = result.images[0]

        # Resize inpainted image to original size
        inpainted_image = output_image.resize(ori_size)

        return inpainted_image
