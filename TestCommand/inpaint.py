# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Image Gen Library
from SD_XL.diffusion_gen import *

# Image Library
from PIL import Image, ImageOps
import cv2

# TRACER 
from TRACER.inference.inference import Inference
from TRACER.config import getConfig, getConfig_Input
from SD_XL.module import *

# Torch and Numpy 
import torch
import numpy as np

# Generate Library
import os
import random
import warnings

# Get args and warning
warnings.filterwarnings("ignore")
# args = getConfig_Input()
args = getConfig()

def main(args):
    # Random Seed
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Some Config Path
    img_url = "./Test_Image/Image.png"
    mask_url = "./Test_Image/Mask.png"
    output_final_url = "./Test_Output/inpaint.png"

    image = Image.open(args.input_path)
    mask = Image.open(args.mask_path)
    prompt = args.prompt 
    negative_prompt = args.negative_prompt
    output_Image = inpaint(image, mask, prompt, negative_prompt)
    output_Image.save(output_final_url)

if __name__ == "__main__":
    main(args)