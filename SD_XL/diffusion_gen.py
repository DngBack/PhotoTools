import torch

import numpy as np

import cv2
from PIL import Image

# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

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
