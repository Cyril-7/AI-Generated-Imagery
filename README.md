# AI-Generated-Imagery
This repository contains code for generating images from prompts using a pre-trained Stable Diffusion model. The model allows for customizable guidance scale, enabling fine-tuning of generated image characteristics.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Modify the configuration parameters in `config.py` to adjust the device, seed, image generation steps, model ID, image size, and guidance scale.
3. Run `generate_image.py` and provide your prompts to generate images. You can specify the number of images to generate for each prompt.

## Example

```python
from config import CFG
from generate_image import generate_image

# Initialize model
image_gen_model = initialize_model()

# Generate images
prompts = ["astronaut in moon"]
generated_images = generate_image(prompts, image_gen_model, num_images=3)

# Display generated images
for i, image in enumerate(generated_images):
    image.show()
