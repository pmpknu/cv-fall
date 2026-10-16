import time

import cv2
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
                    value = int(padded[y + dy, x + dx])
                    minimum = min(minimum, value)

            result[y, x] = minimum

    return result


def erosion_opencv(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(
        image,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def benchmark(func, image, repeats=5):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = func(image)
        times.append(time.perf_counter() - start)

    return result, np.median(times)


def main():
    image = np.zeros((256, 256), dtype=np.uint8)

    # draw black&white figures
    cv2.rectangle(image, (25, 25), (120, 120), 255, -1)
    cv2.circle(image, (180, 160), 50, 255, -1)
    cv2.line(image, (20, 220), (115, 220), 255, 2)

    native_result, native_time = benchmark(erosion_native, image)
    cv_result, cv_time = benchmark(erosion_opencv, image)

    print(f"Native: {native_time * 1000:.3f} ms")
    print(f"OpenCV: {cv_time * 1000:.3f} ms")
    print(f"Results equal: {np.array_equal(native_result, cv_result)}")

    cv2.imwrite("original.png", image)
    cv2.imwrite("native.png", native_result)
    cv2.imwrite("opencv.png", cv_result)


if __name__ == "__main__":
    main()
