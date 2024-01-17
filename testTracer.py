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
    seed = args.seed
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

    # Filename 
    filename = './Test_Output/tracer.png'

    # Get image
    input_url = args.input_path
    inputImage = cv2.imread(input_url)
    save_input = cv2.imwrite(img_url, inputImage)

    # Remove Back ground and get
    arch = "7"
    exp_num = 0
    save_path = os.path.join(
        "results/", "custom_dataset/", f"TE{arch}_{str(exp_num)}"
    )

    # Get pre-mask
    mask_of_image, object_of_image = Inference(args, save_path).test()
    rgb_image = cv2.cvtColor(mask_of_image, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(rgb_image)
    cv2.imwrite(filename, object_of_image) 

if __name__ == "__main__":
    main(args)