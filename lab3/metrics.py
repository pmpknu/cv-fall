import os
from pathlib import Path

os.environ.setdefault(
    "TORCH_HOME", str(Path(__file__).resolve().parent / ".cache" / "torch")
)
os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface")
)

import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision.transforms.functional import to_tensor


def load_image(path):
    with Image.open(path) as image:
        return to_tensor(image.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def main():
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).eval()
    clip_iqa = CLIPImageQualityAssessment(
        model_name_or_path="clip_iqa", data_range=1.0
    ).eval()
    zero_checks = []

    print("degradation   mask       PSNR    LPIPS  CLIP-IQA")
    for degradation in ("downsampling", "noise", "blur"):
        for mask in ("zero", "one", "mixed"):
            scores = []
            for name in ("lena", "baboon", "orange"):
                directory = Path("results") / name / degradation
                reference = load_image(directory / "ground_truth.png")
                output = load_image(directory / f"output_{mask}.png")
                if mask == "zero":
                    with Image.open(directory / "input.png") as image:
                        bicubic = image.convert("RGB").resize(
                            (256, 256), Image.Resampling.BICUBIC
                        )
                    zero_checks.append(
                        torch.equal(output, to_tensor(bicubic).unsqueeze(0))
                    )
                for metric in (psnr, lpips, clip_iqa):
                    metric.reset()
                scores.append(
                    (
                        psnr(output, reference).item(),
                        lpips(output, reference).item(),
                        clip_iqa(output).item(),
                    )
                )
            average = torch.tensor(scores).mean(dim=0)
            print(
                f"{degradation:<13} {mask:<7} {average[0]:8.3f} {average[1]:8.4f} {average[2]:9.4f}"
            )
    print(f"All-zero preservation: {sum(zero_checks)}/{len(zero_checks)}")


if __name__ == "__main__":
    main()
