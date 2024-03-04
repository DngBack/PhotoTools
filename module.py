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
import requests
import json 
import requests
import base64
from io import BytesIO
from PIL import Image


#load module 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diffusion_gen = DiffusionGenerationV2(device=device)
# diffusion_gen.load_module()
diffusion_gen = DiffusionGenerationAPI(device=device)


key = 'KwZbudwQD9xXiWqQzmeXdgtBkVK5X5DMYI3JgxenBahHBTd7oq7IB5x0TRpw'
url = "https://stablediffusionapi.com/api/v4/dreambooth/inpaint"
url_fetch = "https://stablediffusionapi.com/api/v4/dreambooth/fetch"


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


def decode_base64_to_np_array(base64_string):
    try:
        # Decode base64 to binary
        decoded_bytes = base64.b64decode(base64_string)
        
        # Convert binary data to a NumPy array
        image = Image.open(BytesIO(decoded_bytes))
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def get_image_from_url_base64(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            content = response.text
            img = decode_base64_to_np_array(content)
            return img
        else:
            return None
    except Exception as e:
        return None


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

    # Generate Image
    output_Image = diffusion_gen.inpaint_image(image=image, mask=mask, prompt= prompt, negative_prompt=negative_prompt)
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


def bgChangingAPI(image, prompt, negative_prompt):
    """
    Args: 
        image (CV2 Image): Image input for processing 
        prompt (String): Description to be able to change the background of the input image
        negative_prompt (String): The description for the model to generate avoids the following requirements
    Output: 
        Image (CV2 Image)
    """
    # Set Some Config Path
    img_url = "./TRACER/data/custom_dataset/Image.png"

    # Get image
    cv2.imwrite(img_url, image)

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
    mask = ImageOps.invert(mask)

    # Chuyển đổi ảnh sang dạng RGB.
    rgb_image = mask.convert('RGB')

    # Lấy dữ liệu ảnh dưới dạng mảng NumPy.
    cv2_image = np.array(rgb_image)

    # BGR là định dạng màu mặc định của OpenCV.
    mask = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

    # Convert image and mask to base64 
    base64_img = convert_to_base64(image)
    base64_mask = convert_mask_to_base64(mask)

    # Get height and width of image 
    height,width = image.shape[:2]

    # Get API to image 
    API_img_url = diffusion_gen.inpaint_image(image=base64_img, mask=base64_mask, width=width, height=height, prompt=prompt, negative_prompt=negative_prompt, url=url, url_fetch=url_fetch, key=key)

    # Get Image 
    image = get_image_from_url_base64(API_img_url)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image 

def inpaintAPI(image, mask, prompt, negative_prompt):
    # Convert image and mask to base64 
    base64_img = convert_to_base64(image)
    base64_mask = convert_mask_to_base64(mask)

    # Get height and width of image 
    height,width = image.shape[:2]
    # Get API to image 
    API_img_url = diffusion_gen.inpaint_image(image=base64_img, mask=base64_mask, width=width, height=height, url=url, url_fetch=url_fetch, key=key)

    # Get Image 
    image = get_image_from_url_base64(API_img_url)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image