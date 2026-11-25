from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from main import load_model, super_resolution


def degrade(image):
    downsampling = image.resize((64, 64), Image.Resampling.BICUBIC)
    pixels = np.asarray(downsampling, dtype=np.float32)
    noise = np.random.default_rng(42).normal(0, 10, pixels.shape)
    noisy = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    blur = image.filter(ImageFilter.GaussianBlur(radius=2)).resize(
        (64, 64), Image.Resampling.BICUBIC
    )
    return {"downsampling": downsampling, "noise": noisy, "blur": blur}


def main():
    pipeline = load_model()
    masks = {
        "zero": np.zeros((64, 64), dtype=np.uint8),
        "one": np.ones((64, 64), dtype=np.uint8),
        "mixed": np.zeros((64, 64), dtype=np.uint8),
    }
    masks["mixed"][16:48, 16:48] = 1
    mask_directory = Path("results/masks")
    mask_directory.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        Image.fromarray(mask * 255).save(mask_directory / f"{name}.png")

    for name in ("lena", "baboon", "orange"):
        with Image.open(f"assets/{name}.jpg") as image:
            ground_truth = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        for degradation, low_resolution in degrade(ground_truth).items():
            directory = Path("results") / name / degradation
            directory.mkdir(parents=True, exist_ok=True)
            ground_truth.save(directory / "ground_truth.png")
            low_resolution.save(directory / "input.png")
            for mask_name, mask in masks.items():
                output = super_resolution(pipeline, low_resolution, mask)
                output.save(directory / f"output_{mask_name}.png")


if __name__ == "__main__":
    main()
