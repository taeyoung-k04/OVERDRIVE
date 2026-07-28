#!/usr/bin/env python3
"""Align a live camera with a reference frame and preview its BEV image."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils.perspective import apply_new_perspective, make_new_perspective_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "dataset" / "parking" / "frames"
REFERENCE_NAMES = ("Left_Back", "Left_Front", "Right_Back", "Right_Front")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=REFERENCE_NAMES, help="Reference direction. If omitted, an interactive prompt is shown.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=960, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=540, help="Requested camera height.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Reference opacity from 0.0 to 1.0.")
    parser.add_argument("--flip", action="store_true", help="Flip the live camera horizontally before alignment and BEV conversion.")
    return parser.parse_args()


def choose_reference(initial: str | None) -> str:
    if initial is not None:
        return initial

    print("Select a reference frame:")
    for index, name in enumerate(REFERENCE_NAMES, start=1):
        print(f"  {index}: {name}")

    while True:
        answer = input("Reference [1-4]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(REFERENCE_NAMES):
            return REFERENCE_NAMES[int(answer) - 1]
        print("Enter a number from 1 to 4.")


def load_reference(frames_root: Path, name: str) -> np.ndarray:
    path = frames_root / name / "frame_000000s.jpg"
    reference = cv2.imread(str(path))
    if reference is None:
        raise FileNotFoundError(f"Could not read reference image: {path}")
    return reference


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    backends = (
        (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        if sys.platform.startswith("win")
        else (cv2.CAP_ANY,)
    )
    for backend in backends:
        capture = cv2.VideoCapture(args.camera, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if capture.isOpened():
            return capture
        capture.release()
    raise RuntimeError(f"Could not open camera index {args.camera}")


def fit_reference(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frame_size = (frame.shape[1], frame.shape[0])
    if (reference.shape[1], reference.shape[0]) == frame_size:
        return reference
    return cv2.resize(reference, frame_size, interpolation=cv2.INTER_AREA)


def draw_label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_fps(image: np.ndarray, fps: float) -> None:
    cv2.putText(
        image,
        f"FPS: {fps:4.1f}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_reference_status(image: np.ndarray, name: str, alpha: float) -> None:
    cv2.putText(
        image,
        f"{name} | reference {alpha:.0%}",
        (12, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = target_height / image.shape[0]
    size = (
        max(1, round(image.shape[1] * scale)),
        target_height,
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0.0 and 1.0")
    reference_name = choose_reference(args.view)
    reference = load_reference(DEFAULT_FRAMES_ROOT, reference_name)
    capture = open_camera(args)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera}: {actual_width}x{actual_height} "
        f"@ {actual_fps:.1f} FPS"
    )
    print(
        "Controls:\n"
        "  1: use Left_Back/frame_000000s.jpg\n"
        "  2: use Left_Front/frame_000000s.jpg\n"
        "  3: use Right_Back/frame_000000s.jpg\n"
        "  4: use Right_Front/frame_000000s.jpg\n"
        "  -: decrease reference opacity\n"
        "  +: increase reference opacity\n"
        "  ESC: quit"
    )

    window_name = "Camera Alignment | Current Perspective (BEV)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    alpha = args.alpha
    config = None
    fitted_reference = None
    previous_shape = None
    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")
            if args.flip:
                frame = cv2.flip(frame, 1)

            if frame.shape != previous_shape:
                config = make_new_perspective_config(frame.shape)
                fitted_reference = fit_reference(reference, frame)
                previous_shape = frame.shape

            overlay = cv2.addWeighted(
                frame,
                1.0 - alpha,
                fitted_reference,
                alpha,
                0.0,
            )

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                smoothed_fps = (
                    0.9 * smoothed_fps + 0.1 * instant_fps
                    if smoothed_fps
                    else instant_fps
                )

            draw_fps(overlay, smoothed_fps)
            draw_reference_status(overlay, reference_name, alpha)
            bev = apply_new_perspective(frame, config)
            bev_preview = resize_to_height(bev, frame.shape[0])
            draw_label(bev_preview, f"Live BEV | full output {bev.shape[1]}x{bev.shape[0]}")
            preview = cv2.hconcat((overlay, bev_preview))

            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if ord("1") <= key <= ord("4"):
                reference_name = REFERENCE_NAMES[key - ord("1")]
                reference = load_reference(DEFAULT_FRAMES_ROOT, reference_name)
                fitted_reference = fit_reference(reference, frame)
                print(f"Reference: {reference_name}")
            elif key == ord("-"):
                alpha = max(0.0, alpha - 0.05)
            elif key in (ord("+"), ord("=")):
                alpha = min(1.0, alpha + 0.05)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
