#!/usr/bin/env python3
"""Show real-time classified lane semantic overlay from a camera."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from infer_sem_class import make_class_overlay, semantic_to_class_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
        help="Path to the trained semantic-class YOLO weights.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--width", type=int, default=960, help="Camera/frame width. 0 keeps camera default.")
    parser.add_argument("--height", type=int, default=540, help="Camera/frame height. 0 keeps camera default.")
    parser.add_argument(
        "--no-force-size",
        action="store_true",
        help="Do not resize frames when the camera ignores the requested size.",
    )
    parser.add_argument("--flip", action="store_true", help="Horizontally flip camera frames before inference.")
    parser.add_argument("--show-fps", action="store_true", help="Draw measured FPS on the overlay.")
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return capture


def normalize_frame_size(frame, width: int, height: int, force_size: bool):
    if not force_size or width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in the active environment. "
            "Install it before running real-time lane inference."
        ) from exc

    if not args.weights.exists():
        raise SystemExit(f"Weights file does not exist: {args.weights}")

    model = YOLO(str(args.weights), task="semantic")
    capture = open_camera(args.camera, args.width, args.height)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height}",
        flush=True,
    )

    window_name = "Lane Semantic Class Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")

            if args.flip:
                frame = cv2.flip(frame, 1)
            frame = normalize_frame_size(frame, args.width, args.height, not args.no_force_size)

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                task="semantic",
                verbose=False,
            )
            class_map = semantic_to_class_map(results[0].semantic_mask, frame.shape[:2])
            overlay = make_class_overlay(frame, class_map)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            if args.show_fps:
                draw_fps(overlay, fps)

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
