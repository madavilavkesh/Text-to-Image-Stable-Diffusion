````md
---
title: Gen AI Text-to-Image Generator
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.27.0"
app_file: app.py
pinned: false
---

# Gen AI Text-to-Image Generator using Stable Diffusion

This project is a **Text-to-Image Generation Web App** built using **Stable Diffusion**, **Gradio**, and **Diffusers**.  
It allows users to generate AI-powered images from text descriptions with various customizable parameters.

## Features

- **Text-to-Image Generation**: Enter a text prompt to generate AI-generated images.
- **Negative Prompting**: Specify what elements should be avoided in the image.
- **Multiple Scheduler Algorithms**: Supports **Euler Discrete, LMS Discrete, PNDM, DPM Solver Multistep, and DDIM**.
- **Customizable Parameters**:
  - **Inference Steps**: Control the number of denoising steps.
  - **Guidance Scale**: Adjust how closely the image follows the text prompt.
  - **Image Dimensions**: Set the height and width of the generated image.
  - **Batch Size**: Generate multiple images at once.
  - **Seed**: Ensure reproducibility by setting a seed value.
- **User-Friendly Gradio UI** for easy interaction.

## Installation

To run this project locally, install the required dependencies:

```sh
pip install transformers diffusers torch pillow gradio
```
````

## Usage

Run the application using:

```sh
python app.py
```

Then, access the Gradio web interface to generate images.

## License

This project is licensed under the **MIT License**.

```

```
