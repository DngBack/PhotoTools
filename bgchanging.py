# Huggingface: Stable Diffusion Library
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Image Gen Library
from SD_XL.diffusion_gen import *
from SD_XL.post_process import *

# Image Library
from PIL import Image, ImageOps
import cv2

# TRACER and module 
from TRACER.inference.inference import Inference
from TRACER.config import getConfig, getConfig_Input
from module import *

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
    img_url = "./TRACER/data/custom_dataset/Image.png"
    output_final_url = "./Test_Output/bg_output.png"

    # Get image
    input_url = args.input_path
    image = cv2.imread(input_url)
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    output_final = bgChangingAPI(image, prompt, negative_prompt)

    cv2.imwrite(output_final_url, output_final)


if __name__ == "__main__":
    main(args)
