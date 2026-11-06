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


def valid_corners(corners, frame_shape):
    height, width = frame_shape[:2]
    return (
        np.isfinite(corners).all()
        and cv2.isContourConvex(corners)
        and cv2.contourArea(corners) >= 0.001 * width * height
    )


def find_object(reference_keypoints, keypoints, matches, reference_shape):
    if len(matches) < 4:
        return None, None, 0

    src = np.float32([reference_keypoints[m.queryIdx].pt for m in matches])
    dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    inliers = int(mask.sum()) if mask is not None else 0
    if matrix is None or inliers < 8:
        return None, None, inliers

    height, width = reference_shape[:2]
    corners = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix)
    if not valid_corners(transformed, reference_shape):
        return None, None, inliers

    # Inliers match points inside the full reference image of the object.
    points = np.unique(dst[mask.ravel() == 1], axis=0).reshape(-1, 1, 2)
    return transformed, points, inliers


def track_points(previous_gray, gray, points, corners):
    if points is None or len(points) < 8:
        return None, None, 0

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3
    )
    if next_points is None:
        return None, None, 0
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, next_points, None, winSize=(21, 21), maxLevel=3
    )
    if back_points is None:
        return None, None, 0

    # Use the same mask for old and new points to preserve correspondences.
    error = np.linalg.norm(back_points - points, axis=2).ravel()
    good = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < 1.0)
    src, dst = points[good], next_points[good]
    if len(src) < 8:
        return None, None, 0

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if matrix is None or inliers < 8:
        return None, None, inliers
    transformed = cv2.perspectiveTransform(corners, matrix)
    if not valid_corners(transformed, gray.shape):
        return None, None, inliers

    return transformed, dst[mask.ravel() == 1], inliers


def credible_correction(flow, sift, inliers, match_count):
    if inliers < 12 or inliers < 0.5 * match_count:
        return False
    # Strong reference evidence can correct accumulated flow drift.
    if inliers >= 20:
        return True

    # With fewer inliers, also require agreement with the current flow estimate.
    intersection, _ = cv2.intersectConvexConvex(flow, sift)
    union = cv2.contourArea(flow) + cv2.contourArea(sift) - intersection
    diagonal = np.linalg.norm(np.ptp(flow.reshape(4, 2), axis=0))
    largest_error = np.linalg.norm(flow - sift, axis=2).max()
    return intersection >= 0.6 * union and largest_error <= 0.2 * diagonal


def draw_object(frame, corners):
    corners = np.int32(corners)
    cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
    x, y = corners[0, 0]
    cv2.putText(frame, "Mona Lisa", (int(x), max(20, int(y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(
        description="Locate Mona Lisa using SIFT or Lucas-Kanade tracking with SIFT drift correction."
    )
    parser.add_argument("input", type=Path, help="path to the input video")
    parser.add_argument("--method", choices=("sift", "hybrid"), default="hybrid",
                        help="tracking method (default: hybrid)")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input video: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    output = Path("results")
    output.mkdir(exist_ok=True)
    output_stem = f"{args.input.stem}_{args.method}"
    output_path = output / f"{output_stem}.avi"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )

    sift = cv2.SIFT_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    example_frames = {max(2, frame_count * part // 4) for part in (1, 2, 3)}
    processed = 0
    points, corners = None, None

    while True:
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if processed == 0:
            reference = frame.copy()
            reference_keypoints, reference_descriptors = detect_features(reference, sift)
        else:
            keypoints, descriptors = detect_features(frame, sift)
            good_matches = match_features(reference_descriptors, descriptors, matcher)
            sift_corners, sift_points, inliers = find_object(
                reference_keypoints, keypoints, good_matches, reference.shape
            )
            if args.method == "sift":
                corners, points = sift_corners, sift_points
            else:
                if corners is not None:
                    corners, points, _ = track_points(previous_gray, gray, points, corners)
                if corners is None:
                    corners, points = sift_corners, sift_points
                elif sift_corners is not None and credible_correction(
                    corners, sift_corners, inliers, len(good_matches)
                ):
                    corrected = 0.8 * corners + 0.2 * sift_corners
                    if valid_corners(corrected, frame.shape):
                        # Keep the measured flow points; correct only the quadrilateral.
                        corners = corrected

        if corners is not None:
            draw_object(frame, corners)
        writer.write(frame)
        previous_gray = gray
        processed += 1
        if processed in example_frames:
            cv2.imwrite(str(output / f"{output_stem}_frame_{processed:04d}.jpg"), frame)

    capture.release()
    writer.release()
    print(f"Frames processed: {processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
