import requests
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image


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

url = "https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/0-796fea15-8a7e-47b4-b9c7-fc03df88f520.base64"

image = get_image_from_url_base64(url)