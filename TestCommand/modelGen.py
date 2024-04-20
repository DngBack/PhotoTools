# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Image Gen Library
from SD_XL.diffusion_gen import *

# Image Library
from PIL import Image, ImageOps
import cv2

# Library 
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import some Python Library 
from collections import OrderedDict
from clothSeg.options import opt
from clothSeg.process import *

# import get var func
import argparse

def main(args): 
    # set some hyperparameter
    hp_dict = {
        "seed": -305,
        "kernel_size": (5, 5),
        "kernel_iterations": 15,
        "num_inference_steps": 70,
        "denoising_start": 0.70,
        "guidance_scale": 7.5,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_seg_model('clothSeg/model/cloth_segm.pth', device=device)

    palette = get_palette(4)

    img = Image.open(args.input_path).convert('RGB')

    cloth_seg = generate_mask(img, net=model, palette=palette, device=device)
    thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    mask = cloth_seg.convert('L').point(fn, mode='1')
    

    # Model Pipeline calling
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )

    # Execute
    diffusion_gen = DiffusionGenerationV2(inpaint_pipe, hp_dict, device)

    # Get input
    image = Image.open(args.input_path)
    

    # Generate Image
    output_Image = diffusion_gen.inpaint_image(image=image, mask=ImageOps.invert(mask))

    # Save Image
    output_final_url = "./Test_Output/modelGen.png"
    output_Image.save(output_final_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, help="Prompt to generate image")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt", default="disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w, person, draw, art")
    args = parser.parse_args()

    main(args)



    

    