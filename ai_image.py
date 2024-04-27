from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9

def generate_image(prompts, model, num_images=1, guidance_scale=CFG.image_gen_guidance_scale):
    images = []
    for prompt in prompts:
        image = model(
            prompt, num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=guidance_scale
        ).images[0]
        image = image.resize(CFG.image_gen_size)
        images.append(image)
    return images

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

prompts = ["sunset over the ocean"]
generated_images = generate_image(prompts, image_gen_model, num_images=3)

for i, image in enumerate(generated_images):
    image.show()


