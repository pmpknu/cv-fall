import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Convert a video to grayscale and save it as results/<input_stem>.avi."
    )
    parser.add_argument("input", type=Path, help="path to the input video")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input video: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"Input: {args.input}")
    print(f"Dimensions: {width} x {height}")
    print(f"FPS: {fps:.6g}")

    output = Path("results")
    output.mkdir(exist_ok=True)
    output_path = output / f"{args.input.stem}.avi"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )

    processed = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        processed += 1

    capture.release()
    writer.release()
    print(f"Frames processed: {processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
