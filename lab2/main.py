import argparse
from pathlib import Path

import cv2


def detect_features(image, orb):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(gray, None)


def match_features(descriptors1, descriptors2, matcher):
    if (
        descriptors1 is None
        or len(descriptors1) == 0
        or descriptors2 is None
        or len(descriptors2) < 2
    ):
        return []

    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            best, second = pair
            if best.distance < 0.75 * second.distance:
                good_matches.append(best)

    return good_matches


def main():
    parser = argparse.ArgumentParser(
        description="Detect ORB keypoints and match video frames to the first frame."
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

    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    example_frames = {max(2, frame_count * part // 4) for part in (1, 2, 3)}

    processed = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        keypoints, descriptors = detect_features(frame, orb)
        good_matches = []

        if processed == 0:
            reference = frame.copy()
            reference_keypoints = keypoints
            reference_descriptors = descriptors
            print(f"Reference keypoints: {len(reference_keypoints)}")
        else:
            good_matches = match_features(
                reference_descriptors, descriptors, matcher
            )

        marked_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
        writer.write(marked_frame)
        processed += 1

        if processed in example_frames:
            visualization = cv2.drawMatches(
                reference, reference_keypoints, frame, keypoints, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            image_path = output / f"{args.input.stem}_matches_{processed:04d}.jpg"
            cv2.imwrite(str(image_path), visualization)
            print(
                f"Frame {processed}: {len(keypoints)} keypoints, "
                f"{len(good_matches)} good matches; saved {image_path}"
            )

    capture.release()
    writer.release()
    print(f"Frames processed: {processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
