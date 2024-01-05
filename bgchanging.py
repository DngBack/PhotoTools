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
    output_final_url = "./TRACER/output_final/output_final.png"

    # Get image
    input_url = args.input_path
    inputImage = cv2.imread(input_url)
    save_input = cv2.imwrite(img_url, inputImage)

    # Remove Back ground and get
    save_path = os.path.join(
        args.model_path, args.dataset, f"TE{args.arch}_{str(args.exp_num)}"
    )

    # Get pre-mask
    mask_of_image, object_of_image = Inference(args, save_path).test()
    rgb_image = cv2.cvtColor(mask_of_image, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(rgb_image)

    # Setup hyper parameters
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

    # Model Pipeline calling
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Execute
    diffusion_gen = DiffusionGenerationV2(inpaint_pipe, hp_dict, device)

    # Get input
    image = Image.open(img_url)

    # Generate Image
    output_Image = diffusion_gen.inpaint_image(image=image, mask=ImageOps.invert(mask))

    # # Execute
    post_processing = PostProcessing(image, mask, output_Image)
    output_final = post_processing.overlay_object2output()
    output_final.save(output_final_url)


if __name__ == "__main__":
    main(args)
