import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface"))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import (
    LDMSuperResolutionPipeline,
)
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


if __name__ == "__main__":
    main()
