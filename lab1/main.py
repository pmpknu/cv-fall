import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def erosion_native(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    padded = np.pad(image, 1, constant_values=255)
    result = np.empty_like(image)

    for y in range(height):
        for x in range(width):
            minimum = 255
            for dy in range(3):
                for dx in range(3):
                    minimum = min(minimum, int(padded[y + dy, x + dx]))
            result[y, x] = minimum

    return result


def erosion_numpy(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    padded = np.pad(image, 1, constant_values=255)
    result = np.full_like(image, 255)

    for dy in range(3):
        for dx in range(3):
            window = padded[dy : dy + height, dx : dx + width]
            np.minimum(result, window, out=result)

    return result


def erosion_opencv(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(image, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)


def benchmark(func, image, repeats=20):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(image)
        times.append(time.perf_counter() - start)

    return result, float(np.median(times))


def main():
    image = cv2.imread("assets/orange.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit("image does not exist")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    native, native_time = benchmark(erosion_native, binary)
    numpy_result, numpy_time = benchmark(erosion_numpy, binary)
    opencv, opencv_time = benchmark(erosion_opencv, binary)

    equal = np.array_equal(native, numpy_result) and np.array_equal(native, opencv)
    print(f"Results equal: {equal}")

    names = ["Python (naive)", "NumPy", "OpenCV"]
    times_ms = [native_time * 1000, numpy_time * 1000, opencv_time * 1000]
    for name, elapsed in zip(names, times_ms):
        print(f"{name}: {elapsed:.3f} ms")
    print(f"NumPy speedup: {native_time / numpy_time:.1f}x")
    print(f"OpenCV speedup: {native_time / opencv_time:.1f}x")

    output = Path("results")
    output.mkdir(exist_ok=True)
    images = {
        "original": image,
        "binary": binary,
        "native": native,
        "numpy": numpy_result,
        "opencv": opencv,
    }
    figure, axes = plt.subplots(1, len(images), figsize=(16, 4))
    for axis, (name, result) in zip(axes, images.items()):
        cv2.imwrite(str(output / f"{name}.png"), result)
        axis.imshow(result, cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
