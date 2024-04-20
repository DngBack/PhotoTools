from SD_XL.module import bgChanging
from PIL import Image
import gradio as gr


# bgChanging(image, prompt, negative_prompt)
def Test_bgChanging(image, prompt, negative_prompt): 
    return bgChanging(image, prompt, negative_prompt)


gr.Interface(
    bgChanging,
    title = 'Stable Diffusion In-Painting Tool on Colab with Gradio',
    inputs=[
        gr.Image(type = 'pil'),
        gr.Textbox(label = 'prompt'),
        gr.Textbox(label = 'negative_prompt')
    ],
    outputs = [
        gr.Image()
        ]
).launch(debug = True)
