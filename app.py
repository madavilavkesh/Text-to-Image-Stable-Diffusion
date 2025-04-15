!pip install transformers diffusers torch pillow gradio

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, LMSDiscreteScheduler, \
    PNDMScheduler, DPMSolverMultistepScheduler, DDIMScheduler
import torch
import gradio as gr
from PIL import Image

model_id = "stabilityai/stable-diffusion-2-1"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
sd_pipeline.to("cuda")

schedulers = {
    "Euler Discrete": EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
    "LMS Discrete": LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
    "PNDM": PNDMScheduler.from_pretrained(model_id, subfolder="scheduler"),
    "DPM Solver Multistep": DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler"),
    "DDIM": DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
}

def generate_image(
    prompt,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    batch_size=1,
    seed=None,
    scheduler="Euler Discrete",
):
    sd_pipeline.scheduler = schedulers.get(scheduler, sd_pipeline.scheduler)
    generator = torch.manual_seed(seed) if seed else None
    images = sd_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_images_per_prompt=batch_size,
        generator=generator
    ).images
    return images

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Slider(step=1, minimum=1, maximum=100, value=50, label="Num Inference Steps"),
        gr.Slider(step=0.1, minimum=1, maximum=20, value=7.5, label="Guidance Scale"),
        gr.Slider(step=64, minimum=256, maximum=1024, value=512, label="Height"),
        gr.Slider(step=64, minimum=256, maximum=1024, value=512, label="Width"),
        gr.Slider(step=1, minimum=1, maximum=4, value=1, label="Batch Size"),
        gr.Number(label="Seed"),
        gr.Dropdown(
            choices=["Euler Discrete", "LMS Discrete", "PNDM", "DPM Solver Multistep", "DDIM"],
            value="Euler Discrete",
            label="Scheduler Algorithm"
        ),
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="Gen AI Text-to-Image Generator using Stable Diffusion",
    description="Generate AI images from text with customizable settings.",
)

interface.launch(share=True, debug=True)
