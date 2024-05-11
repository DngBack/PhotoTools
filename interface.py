import gradio as gr
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "stablediffusionapi/epicdream"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

def transform (init_image, textPrompt, strength=0.5, guidance_scale=15):
  init_image = Image.open(init_image).convert("RGB")
  init_image = init_image.resize((768, 512))
  images = pipe(prompt=textPrompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
  image = images[0]
  return image

demo = gr.Interface(
    fn=transform,
    inputs=[gr.Image(type='filepath'), "text", gr.Slider(0,1), gr.Slider(1,30)],
    outputs=["image"],
    allow_flagging="never"
)
demo.launch()