from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from SD_XL.module import bgChanging, inpaint 
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

def base64_to_pil(im_b64):
    im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object
    return img

def pil_to_base64(img):
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


class ImageInfoBgChanging(BaseModel):
    """
    Set pydantic
        Image: Base64 Image 
        prompt: string 
        negative_prompt: string
    """
    image: str
    prompt: str
    negative_prompt: str

class ImageInfoInpaint(BaseModel):
    """
    Set pydantic
        Image: Base64 Image
        Mask: Base64 Image
        prompt: string 
        negative_prompt: string
    """
    image: str
    mask: str
    prompt: str 
    negative_prompt: str

@app.post("/bg_changing")
def bg_Changing(imageInfo: ImageInfoBgChanging): 
    image = imageInfo.image
    prompt = imageInfo.prompt
    negative_prompt = imageInfo.negative_prompt

    # Convert base64 to image
    image = base64.b64decode(image)
    image = BytesIO(image)
    image = Image.open(image)
    image = bgChanging(image, prompt, negative_prompt)

    # Convert PIL image to base64 string
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")

    return img_str


@app.post("/inpainting")
def inpainting(imageInfo:ImageInfoInpaint):
    image = imageInfo.image
    mask = imageInfo.mask
    prompt = imageInfo.prompt
    negative_prompt = imageInfo.negative_prompt

    # Convert base64 to pil 
    image = base64_to_pil(image)
    mask = base64_to_pil(mask)

    # Inpainting \
    image_output = inpaint(image, mask, prompt, negative_prompt)

    # Convert base64 to pil 
    image_output = pil_to_base64(image_output)

    return image_output

if __name__ == "__main__":
    uvicorn.run("main:app", port=8081, reload=True)
