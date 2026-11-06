import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_features(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


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


def find_object(reference_keypoints, keypoints, matches, reference_shape):
    if len(matches) < 4:
        return None, 0

    src = np.float32([reference_keypoints[m.queryIdx].pt for m in matches])
    dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    inliers = int(mask.sum()) if mask is not None else 0
    if matrix is None or inliers < 8:
        return None, inliers

    height, width = reference_shape[:2]
    corners = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix)
    return np.int32(transformed), inliers


def main():
    parser = argparse.ArgumentParser(
        description="Track a planar object using sift and RANSAC."
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

    sift = cv2.SIFT_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    example_frames = {max(2, frame_count * part // 4) for part in (1, 2, 3)}

    processed = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        keypoints, descriptors = detect_features(frame, sift)
        good_matches = []
        corners, inliers = None, 0

        if processed == 0:
            reference = frame.copy()
            reference_keypoints = keypoints
            reference_descriptors = descriptors
            print(f"Reference keypoints: {len(reference_keypoints)}")
        else:
            good_matches = match_features(
                reference_descriptors, descriptors, matcher
            )
            corners, inliers = find_object(
                reference_keypoints, keypoints, good_matches, reference.shape
            )

        if corners is not None:
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            x, y = corners[0, 0]
            cv2.putText(frame, "Mona Lisa", (int(x), max(20, int(y) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)
        processed += 1

        if processed in example_frames:
            visualization = cv2.drawMatches(
                reference, reference_keypoints, frame, keypoints, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            image_path = output / f"{args.input.stem}_matches_{processed:04d}.jpg"
            cv2.imwrite(str(image_path), visualization)
            cv2.imwrite(str(output / f"{args.input.stem}_frame_{processed:04d}.jpg"), frame)
            print(
                f"Frame {processed}: {len(keypoints)} keypoints, "
                f"{len(good_matches)} good matches, {inliers} inliers, "
                f"detected={corners is not None}; saved {image_path}"
            )

    capture.release()
    writer.release()
    print(f"Frames processed: {processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
