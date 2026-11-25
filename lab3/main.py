from pathlib import Path

from PIL import Image


def main():
    with Image.open("assets/test.jpg") as image:
        ground_truth = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    low_resolution = ground_truth.resize((64, 64), Image.Resampling.BICUBIC)
    bicubic = low_resolution.resize((256, 256), Image.Resampling.BICUBIC)

    Path("results").mkdir(exist_ok=True)
    ground_truth.save("results/ground_truth.png")
    low_resolution.save("results/input.png")
    bicubic.save("results/bicubic.png")


if __name__ == "__main__":
    main()
