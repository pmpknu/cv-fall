import os
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface")
)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import (
    LDMSuperResolutionPipeline,
    preprocess,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(
        "CompVis/ldm-super-resolution-4x-openimages",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    for component in (pipeline.unet, pipeline.vqvae):
        for module in component.modules():
            if isinstance(module, Attention):
                module.set_attention_slice(1)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


@torch.inference_mode()
def super_resolution(pipeline, image):
    # prepare
    condition = preprocess(image)
    height, width = condition.shape[-2:]
    dtype = next(pipeline.unet.parameters()).dtype

    # initial noise
    generator = torch.Generator("cpu").manual_seed(42)
    latent_shape = (1, pipeline.unet.config.in_channels // 2, height, width)
    latents = randn_tensor(
        latent_shape, generator=generator, device=pipeline.device, dtype=dtype
    )
    condition = condition.to(device=pipeline.device, dtype=dtype)
    pipeline.scheduler.set_timesteps(20, device=pipeline.device)
    latents = latents * pipeline.scheduler.init_noise_sigma

    # diffusion
    for timestep in pipeline.scheduler.timesteps:
        model_input = torch.cat([latents, condition], dim=1)
        # predict noise
        model_input = pipeline.scheduler.scale_model_input(model_input, timestep)
        predicted_noise = pipeline.unet(model_input, timestep).sample
        # remove noise
        latents = pipeline.scheduler.step(
            predicted_noise, timestep, latents, eta=0
        ).prev_sample

    # latents to image
    decoded = pipeline.vqvae.decode(latents).sample
    decoded = decoded.clamp(-1, 1) / 2 + 0.5
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
    return pipeline.numpy_to_pil(decoded)[0]


def main():
    with Image.open("assets/test.jpg") as image:
        ground_truth = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    low_resolution = ground_truth.resize((64, 64), Image.Resampling.BICUBIC)
    bicubic = low_resolution.resize((256, 256), Image.Resampling.BICUBIC)

    Path("results").mkdir(exist_ok=True)
    ground_truth.save("results/ground_truth.png")
    low_resolution.save("results/input.png")
    bicubic.save("results/bicubic.png")

    pipeline = load_model()
    with torch.inference_mode():
        output = pipeline(
            image=low_resolution,
            num_inference_steps=20,
            generator=torch.Generator("cpu").manual_seed(42),
        ).images[0]
    output.save("results/sr.png")

    custom_output = super_resolution(pipeline, low_resolution)
    custom_output.save("results/custom_sr.png")
    print("Results identical:", output.tobytes() == custom_output.tobytes())


if __name__ == "__main__":
    main()
