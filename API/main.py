from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from SD_XL.module import bgChanging, inpaint 
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

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
