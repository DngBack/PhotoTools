import torch

import numpy as np

import cv2
from PIL import Image

class DiffusionGenerationV2:
    """
    Stable Diffusion for generation process.
    Using Stable Diffusion 2.0 from stability

    """

    def __init__(self, inpaint_pipe, hp_dict, device):
        """
        Args:
            inpaint_pipe : Stable Diffusion inpaint pipeline. Note that: input size = 512
            refine_pipe : Stable Diffusion refiner pipeline. Note that: input size = 1024
            hp_dict (dict): Hyperparameters dicitionary for generation.
            device (torch.device): Device used.
        """
        # Setup device
        self.device = device

        # Setup pipelines
        self.inpaint_pipe = inpaint_pipe.to(self.device)

        # Setup hyperparameters dictionary
        self.hp_dict = hp_dict

    def inpaint_image(self, image, mask):
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
        generator = torch.Generator(self.device).manual_seed(self.hp_dict["seed"])
        result = self.inpaint_pipe(
            image=input_image,
            mask_image=input_mask,
            prompt=self.hp_dict["prompt"],
            negative_prompt=self.hp_dict["negative_prompt"],
            generator=generator,
        )
        output_image = result.images[0]

        # Resize inpainted image to original size
        inpainted_image = output_image.resize(ori_size)

        return inpainted_image
