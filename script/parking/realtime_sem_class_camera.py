#!/usr/bin/env python3
"""Show a real-time parking semantic overlay from a camera."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

from infer_sem_class import (
    DEFAULT_WEIGHTS,
    load_semantic_model,
    make_overlay,
    semantic_to_class_map,
)
from utils.filter_cars import (
    filter_cars_in_parking_region,
    remove_car_detections,
)
from utils.lane_detect import (
    ParkingDotLineDetector,
    ParkingLineDetector,
    Line,
    ReferenceLineDetector,
    draw_line_points,
    draw_line,
    draw_parking_lines,
)
from utils.phase_control import PhaseController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to .pt or .onnx weights.",
    )
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960, help="Camera/frame width. 0 keeps the camera default.")
    parser.add_argument("--height", type=int, default=540, help="Camera/frame height. 0 keeps the camera default.")
    parser.add_argument("--camera-fps", type=int, default=20, help="Requested camera FPS. 0 keeps the camera default.")
    parser.add_argument("--flip", action="store_true", help="Horizontally flip frames before inference.")
    parser.add_argument("--show-fps", action="store_true", help="Draw measured FPS and capture-to-display delay.")
    return parser.parse_args()


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
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
    """Drain the camera continuously and expose only its newest frame."""

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
            captured_at = time.perf_counter()
            with self.condition:
                if not ok:
                    self.stopped = True
                    self.condition.notify_all()
                    return
                if self.stopped:
                    return
                self.frame = frame
                self.frame_time = captured_at
                self.condition.notify_all()

    def read(self):
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


def normalize_frame_size(frame, width: int, height: int):
    if width <= 0 or height <= 0:
        return frame
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_performance(frame, fps: float, delay: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}  Delay: {delay:.3f}s",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_phase(frame, phase: int) -> None:
    """Draw the current parking phase in the upper-right corner."""
    text = f"PHASE {phase}"
    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2,
    )
    cv2.putText(
        frame,
        text,
        (max(12, frame.shape[1] - text_size[0] - 12), 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_reference_status(frame, line: Line) -> None:
    """Draw reference-line fitting information on the preview."""
    if line.valid:
        text = (
            f"REFERENCE LINE: OK  confidence={line.confidence:.2f}"
        )
        color = (70, 255, 70)
    else:
        text = f"REFERENCE LINE: LOST"
        color = (0, 80, 255)

    cv2.putText(
        frame,
        text,
        (12, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_parking_dot_status(frame, line: Line) -> None:
    """Draw parking-dot-line state directly above reference-line state."""
    if line.valid:
        text = (
            f"PARKING DOT LINE: OK  confidence={line.confidence:.2f}"
        )
        color = (255, 255, 70)
    else:
        text = f"PARKING DOT LINE: LOST"
        color = (0, 80, 255)

    cv2.putText(
        frame,
        text,
        (12, frame.shape[0] - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    if (args.width == 0) != (args.height == 0):
        raise SystemExit("--width and --height must both be 0 or both be positive")
    if args.width < 0 or args.height < 0:
        raise SystemExit("--width and --height cannot be negative")

    model = load_semantic_model(args.weights, args.backend)
    capture = open_camera(
        args.camera,
        args.width,
        args.height,
        args.camera_fps,
    )
    reader = LatestFrameReader(capture)
    print(
        f"Camera {args.camera}: "
        f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS",
        flush=True,
    )

    window_name = "Parking Semantic Class Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    previous_time = time.perf_counter()
    measured_fps = 0.0
    line_detector = ReferenceLineDetector()
    parking_dot_detector = ParkingDotLineDetector()
    parking_line_detector = ParkingLineDetector()
    phase_controller = PhaseController()

    try:
        while True:
            ok, frame, frame_time = reader.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")
            if args.flip:
                frame = cv2.flip(frame, 1)
            frame = normalize_frame_size(frame, args.width, args.height)

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                task="semantic",
                verbose=False,
                stream=True,
            )
            result = next(iter(results))
            class_map = semantic_to_class_map(
                result.semantic_mask,
                frame.shape[:2],
            )
            reference_line = line_detector.detect(class_map)
            parking_dot_line = None
            parking_lines = None
            if phase_controller.phase == 0:
                parking_dot_line = parking_dot_detector.detect(class_map)
                parking_lines = parking_line_detector.detect(
                    class_map,
                    excluded_points=parking_dot_line.rejected_points,
                )
                class_map = filter_cars_in_parking_region(
                    class_map,
                    parking_dot_line,
                    parking_lines,
                )
                phase_controller.update(
                    class_map,
                    parking_lines,
                    parking_dot_line,
                    now=time.perf_counter(),
                )
            elif phase_controller.phase in (1, 2, 3, 4, 5, 6):
                phase_controller.update(
                    class_map,
                    reference_line=reference_line,
                    now=time.perf_counter(),
                )
            if phase_controller.phase >= 1:
                class_map = remove_car_detections(class_map)
            preview = make_overlay(frame, class_map)
            if phase_controller.phase == 0:
                draw_parking_lines(
                    preview,
                    parking_lines,
                    color=(0, 140, 255),
                    thickness=3,
                )
            draw_line(
                preview,
                reference_line,
                color=(0, 255, 0),
                thickness=3,
            )
            if phase_controller.phase == 0:
                draw_line(
                    preview,
                    parking_dot_line,
                    color=(0, 255, 255),
                    thickness=3,
                )
                draw_line_points(
                    preview,
                    parking_dot_line,
                    color=(0, 200, 255),
                    radius=6,
                )
                draw_parking_dot_status(preview, parking_dot_line)
            draw_reference_status(preview, reference_line)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                measured_fps = (
                    0.9 * measured_fps + 0.1 * instant_fps
                    if measured_fps
                    else instant_fps
                )
            if args.show_fps:
                draw_performance(preview, measured_fps, now - frame_time)
            draw_phase(preview, phase_controller.phase)

            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
