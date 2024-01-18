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


def bgChanging(Image, prompt, negative_prompt):
    """
    """

    # Set Some Config Path
    img_url = "./TRACER/data/custom_dataset/Image.png"

    # Get image
    save_input = cv2.imwrite(img_url, image)

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
    hp_dict = {
        "seed": -305,
        "kernel_size": (5, 5),
        "kernel_iterations": 15,
        "num_inference_steps": 70,
        "denoising_start": 0.70,
        "guidance_scale": 7.5,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Pipeline calling
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )

    # Load model 
    diffusion_gen = DiffusionGenerationV2(inpaint_pipe, hp_dict, device)

    # Get input
    image = Image.open(img_url)

    # Generate Image
    output_Image = diffusion_gen.inpaint_image(image=image, mask=ImageOps.invert(mask))

    # # Execute
    post_processing = PostProcessing(image, mask, output_Image)
    output_final = post_processing.overlay_object2output()
    # output_final.save(output_final_url)

    return output_final


def inpaint(image, mask, prompt, negative_prompt):
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

    # Set Some Config Path
    img_url = "./TRACER/data/custom_dataset/Image.png"

    # Get image
    save_input = cv2.imwrite(img_url, image)

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