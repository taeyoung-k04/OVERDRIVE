#!/usr/bin/env python3
"""Show real-time lane, car, and traffic-light semantic overlay from a camera."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from infer_sem_class import (
    DEFAULT_WEIGHTS,
    load_semantic_model,
    make_class_overlay,
    semantic_to_class_map,
)
from utils.postprocess import (
    CLASS_TO_ID,
    add_postprocess_args,
    postprocess_class_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to .pt or .onnx weights. With --backend onnx, a .pt suffix is replaced with .onnx.",
    )
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--width", type=int, default=960, help="Camera/frame width. 0 keeps camera default.")
    parser.add_argument("--height", type=int, default=540, help="Camera/frame height. 0 keeps camera default.")
    parser.add_argument("--camera-fps", type=int, default=20, help="Requested camera FPS. 0 keeps camera default.")
    parser.add_argument("--flip", action="store_true", help="Horizontally flip camera frames before inference.")
    parser.add_argument("--show-fps", action="store_true", help="Draw measured FPS on the overlay.")
    add_postprocess_args(parser)
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        capture.set(cv2.CAP_PROP_FPS, fps)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return capture


class LatestFrameReader:
    """Continuously drain the camera and keep only the newest frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self.condition = threading.Condition()
        self.frame = None
        self.frame_time = 0.0
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            ok, frame = self.capture.read()
            frame_time = time.perf_counter()
            with self.condition:
                if not ok:
                    self.stopped = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.frame_time = frame_time
                self.condition.notify_all()

    def read(self):
        with self.condition:
            if self.frame is None and not self.stopped:
                self.condition.wait(timeout=2.0)
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def read_with_timestamp(self):
        with self.condition:
            if self.frame is None and not self.stopped:
                self.condition.wait(timeout=2.0)
            if self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.frame_time

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)


def normalize_frame_size(frame, width: int, height: int, force_size: bool):
    if not force_size or width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def keep_lane_marking_classes(class_map):
    kept_ids = [
        CLASS_TO_ID[class_name]
        for class_name in ("lane_left", "lane_center", "lane_right", "stop_line")
    ]
    return np.where(np.isin(class_map, kept_ids), class_map, CLASS_TO_ID["background"]).astype(class_map.dtype)


def make_classmap_canvas(class_map, dtype):
    height, width = class_map.shape[:2]
    return np.zeros((height, width, 3), dtype=dtype)


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


def draw_delay(frame, delay_seconds: float, right_edge: int | None = None) -> None:
    text = f"Delay: {delay_seconds:.3f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    margin = 12
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    right_edge = frame.shape[1] if right_edge is None else min(right_edge, frame.shape[1])
    x = max(margin, right_edge - text_size[0] - margin)
    y = margin + text_size[1]
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    model = load_semantic_model(args.weights, args.backend)

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reader = LatestFrameReader(capture)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height} @ {actual_fps:.1f} FPS",
        flush=True,
    )

    window_name = "Lane and Object Semantic Class Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame, frame_time = reader.read_with_timestamp()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")

            if args.flip:
                frame = cv2.flip(frame, 1)
            frame = normalize_frame_size(frame, args.width, args.height, True)

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                task="semantic",
                verbose=False,
                stream=True,
            )
            result = next(iter(results))
            class_map = semantic_to_class_map(result.semantic_mask, frame.shape[:2])
            if args.postprocess:
                class_map = postprocess_class_map(class_map)

            preview = make_class_overlay(frame, class_map)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            if args.show_fps:
                draw_fps(preview, fps)
                draw_delay(preview, now - frame_time)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
