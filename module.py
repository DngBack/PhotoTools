# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Image Gen Library
from SD_XL.diffusion_gen import *
from SD_XL.post_process import *

# Image Library
from PIL import Image, ImageOps
import cv2

# TRACER 
from TRACER.inference.inference import Inference
from TRACER.config import getConfig, getConfig_Input

# Torch and Numpy 
import torch
import numpy as np

# Generate Library
import os

#load module 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion_gen = DiffusionGenerationV2(device=device)
diffusion_gen.load_module()


def bgChanging(image, prompt, negative_prompt):
    """
    Args: 
        image (PIL Image): Image input for processing 
        prompt (String): Description to be able to change the background of the input image
        negative_prompt (String): The description for the model to generate avoids the following requirements
    Output: 
        Image (PIL Image)
    """

    # Set Some Config Path
    img_url = "./TRACER/data/custom_dataset/Image.png"

    # Get image
    image.save(img_url)

    # Setting 
    arch = "7"
    exp_num = 0
    save_path = os.path.join(
        "results/", "custom_dataset/", f"TE{arch}_{str(exp_num)}"
    )

    # Get pre-mask
    mask_of_image, object_of_image = Inference(save_path).test()
    rgb_image = cv2.cvtColor(mask_of_image, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(rgb_image)
    thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    mask = mask.convert('L').point(fn, mode='1')

    # Setup hyper parameters
    # hp_dict = {
    #     "seed": -305,
    #     "kernel_size": (5, 5),
    #     "kernel_iterations": 15,
    #     "num_inference_steps": 70,
    #     "denoising_start": 0.70,
    #     "guidance_scale": 7.5,
    #     "prompt": prompt,
    #     "negative_prompt": negative_prompt,
    # }

    # Model Pipeline calling
    # inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=torch.float32,
    # )


    # Get input
    image = Image.open(img_url)

    # Generate Image
    output_Image = diffusion_gen.inpaint_image(image=image, mask=ImageOps.invert(mask), prompt=prompt, negative_prompt=negative_prompt)

    # # Execute
    post_processing = PostProcessing(image, mask, output_Image)
    output_final = post_processing.overlay_object2output()
    # output_final.save(output_final_url)

    return output_final


def inpaint(image, mask, prompt, negative_prompt):
    """
    Args: 
        image (PIL Image): Image input for processing 
        mask (PIL Image): Area that can be changed in the image
        prompt (String): Description to change the highlighted part
        negative_prompt (String): The description for the model to generate avoids the following
    Output:
        image (PIL Image)
    """
    # Setup hyper parameters
    hp_dict = {
        "seed" : 116,
        "kernel_size": (5,5),
        "kernel_iterations" : 15,
        "num_inference_steps" : 70,
        "denoising_start" : 0.70,
        "guidance_scale" : 7.5,
        "prompt" : prompt,
        "negative_prompt" : negative_prompt,
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Pipeline calling
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )

    # Execute
    diffusion_gen = DiffusionGenerationV2(inpaint_pipe, hp_dict, device)

    # Generate Image
    output_Image = diffusion_gen.inpaint_image(image=image, mask=mask)
    return output_Image 


def rmbg(image):
    """
    Args:
        image (PIL Image): Image input for processing
    Output:
        image (PIL Image)
    """

    # Set Some Config Path
    img_url = "./TRACER/data/custom_dataset/Image.png"

    # Get image
    image.save(img_url)

    # Setting 
    arch = "7"
    exp_num = 0
    save_path = os.path.join(
        "results/", "custom_dataset/", f"TE{arch}_{str(exp_num)}"
    )

    # Get pre-mask
    mask_of_image, object_of_image = Inference(save_path).test()
    rgb_image = cv2.cvtColor(mask_of_image, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(rgb_image)
    return object_of_image