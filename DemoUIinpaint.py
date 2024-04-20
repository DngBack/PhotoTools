# Install Liblaries
import inspect
from typing import List, Optional, Union
import numpy as np
import torch
import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline

#Select device and create a pipe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "stablediffusionapi/epicdream"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
).to(device)

# Prediction Method
def predict(dict, prompt):
  image =  dict['image'].convert("RGB").resize((512, 512))
  mask_image = dict['mask'].convert("RGB").resize((512, 512))
  images = pipe(prompt=prompt, image=image, mask_image=mask_image).images
  return(images)

# Gradio run and implement interface
gr.Interface(
    predict,
    title = 'Stable Diffusion In-Painting',
    inputs=[
        gr.Image(source = 'upload', tool = 'sketch', type = 'pil'),
        gr.Textbox(label = 'prompt')
    ],
    outputs = [
        gr.Image()
        ]
).launch(debug=True,share=True)
