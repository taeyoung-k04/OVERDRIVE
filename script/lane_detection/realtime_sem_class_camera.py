#!/usr/bin/env python3
"""Show real-time classified lane semantic overlay from a camera."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from infer_sem_class import (
    CLASS_TO_ID,
    DEFAULT_WEIGHTS,
    OVERLAY_COLORS,
    add_postprocess_args,
    load_postprocess_config,
    load_semantic_model,
    make_class_overlay,
    postprocess_class_map,
    semantic_to_class_map,
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
    parser.add_argument(
        "--no-force-size",
        action="store_true",
        help="Do not resize frames when the camera ignores the requested size.",
    )
    parser.add_argument(
        "--buffered-camera",
        action="store_true",
        help="Use normal blocking camera reads instead of the low-latency latest-frame reader.",
    )
    parser.add_argument(
        "--preview",
        choices=("overlay", "lines"),
        default="overlay",
        help="Display mode.",
    )
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
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.stopped = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.condition.notify_all()

    def read(self):
        with self.condition:
            if self.frame is None and not self.stopped:
                self.condition.wait(timeout=2.0)
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

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


def draw_fitted_line(preview, points, color, thickness: int) -> None:
    if points is None or len(points) < 24:
        return

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    xs = points[:, 0, 0]
    ys = points[:, 0, 1]
    height, width = preview.shape[:2]

    if abs(vy) > 1e-4:
        y1 = int(np.clip(ys.min(), 0, height - 1))
        y2 = int(np.clip(ys.max(), 0, height - 1))
        x1 = int(np.clip(x0 + (y1 - y0) * vx / vy, 0, width - 1))
        x2 = int(np.clip(x0 + (y2 - y0) * vx / vy, 0, width - 1))
    elif abs(vx) > 1e-4:
        x1 = int(np.clip(xs.min(), 0, width - 1))
        x2 = int(np.clip(xs.max(), 0, width - 1))
        y1 = int(np.clip(y0 + (x1 - x0) * vy / vx, 0, height - 1))
        y2 = int(np.clip(y0 + (x2 - x0) * vy / vx, 0, height - 1))
    else:
        return

    cv2.line(preview, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def make_line_preview(image, class_map):
    preview = make_class_overlay(image, class_map)
    for class_name in ("lane_left", "lane_center", "lane_right", "stop_line"):
        class_id = CLASS_TO_ID[class_name]
        points = cv2.findNonZero((class_map == class_id).astype(np.uint8))
        thickness = 5 if class_name == "stop_line" else 7
        draw_fitted_line(preview, points, OVERLAY_COLORS[class_id], thickness)
    return preview


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
    model = load_semantic_model(args.weights, args.backend)
    postprocess_config = load_postprocess_config(args.postprocess_config) if args.postprocess else None

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reader = None if args.buffered_camera else LatestFrameReader(capture)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height} @ {actual_fps:.1f} FPS",
        flush=True,
    )

    window_name = "Lane Semantic Class Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = reader.read() if reader is not None else capture.read()
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
            if postprocess_config is not None:
                class_map = postprocess_class_map(class_map, postprocess_config)
            if args.preview == "overlay":
                preview = make_class_overlay(frame, class_map)
            else:
                preview = make_line_preview(frame, class_map)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            if args.show_fps:
                draw_fps(preview, fps)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if reader is not None:
            reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
