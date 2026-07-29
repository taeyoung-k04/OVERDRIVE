#!/usr/bin/env python3
"""Render parking videos using the semantic-segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from infer_sem_class import (
    DEFAULT_WEIGHTS,
    load_semantic_model,
    make_overlay,
    semantic_to_class_map,
)
from utils.lane_detect import (
    ParkingDotLineDetector,
    ReferenceLineDetector,
    draw_line_points,
    draw_line,
)


def draw_reference_status(
    image: np.ndarray,
    *,
    valid: bool,
    confidence: float,
    reason: str,
) -> None:
    """Draw the current reference-line fitting state."""
    if valid:
        text = f"REFERENCE LINE: OK  confidence={confidence:.2f}"
        color = (70, 255, 70)
    else:
        text = f"REFERENCE LINE: LOST"
        color = (0, 80, 255)

    cv2.putText(
        image,
        text,
        (12, image.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_parking_dot_status(
    image: np.ndarray,
    *,
    valid: bool,
    confidence: float,
    reason: str,
) -> None:
    """Draw parking-dot-line state directly above reference-line state."""
    if valid:
        text = f"PARKING DOT LINE: OK  confidence={confidence:.2f}"
        color = (255, 255, 70)
    else:
        text = f"PARKING DOT LINE: LOST"
        color = (0, 80, 255)

    cv2.putText(
        image,
        text,
        (12, image.shape[0] - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def process_batch(
    model,
    frames: list[np.ndarray],
    imgsz: int,
    device: str,
    line_detector: ReferenceLineDetector,
    parking_dot_detector: ParkingDotLineDetector,
) -> list[np.ndarray]:
    results = model.predict(
        source=frames,
        imgsz=imgsz,
        device=device,
        task="semantic",
        verbose=False,
    )
    overlays: list[np.ndarray] = []
    for frame, result in zip(frames, results):
        class_map = semantic_to_class_map(
            result.semantic_mask,
            frame.shape[:2],
        )
        reference_line = line_detector.detect(class_map)
        parking_dot_line = parking_dot_detector.detect(class_map)
        overlay = make_overlay(frame, class_map)
        draw_line(
            overlay,
            reference_line,
            color=(0, 255, 0),
            thickness=3,
        )
        draw_line(
            overlay,
            parking_dot_line,
            color=(0, 255, 255),
            thickness=3,
        )
        draw_line_points(
            overlay,
            parking_dot_line,
            color=(0, 200, 255),
            radius=6,
        )
        draw_parking_dot_status(
            overlay,
            valid=parking_dot_line.valid,
            confidence=parking_dot_line.confidence,
            reason=parking_dot_line.reason,
        )
        draw_reference_status(
            overlay,
            valid=reference_line.valid,
            confidence=reference_line.confidence,
            reason=reference_line.reason,
        )
        overlays.append(overlay)
    return overlays


def render_video(
    model,
    source: Path,
    destination: Path,
    output_fps: float,
    imgsz: int,
    device: str,
    batch_size: int,
    width: int,
    height: int,
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or output_fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    force_size = width > 0 and height > 0
    frame_width = width if force_size else source_width
    frame_height = height if force_size else source_height

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {destination}")

    frame_index = 0
    written = 0
    next_time = 0.0
    frame_step_time = 1.0 / output_fps
    batch: list[np.ndarray] = []
    line_detector = ReferenceLineDetector()
    parking_dot_detector = ParkingDotLineDetector()
    progress = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=source.stem,
        unit="frame",
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            progress.update()
            time_seconds = frame_index / source_fps
            frame_index += 1
            if time_seconds + 1e-9 < next_time:
                continue

            if force_size and frame.shape[:2] != (frame_height, frame_width):
                frame = cv2.resize(
                    frame,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_AREA,
                )
            batch.append(frame)
            next_time += frame_step_time

            if len(batch) >= batch_size:
                for overlay in process_batch(
                    model,
                    batch,
                    imgsz,
                    device,
                    line_detector,
                    parking_dot_detector,
                ):
                    writer.write(overlay)
                    written += 1
                batch.clear()
                progress.set_postfix(written=written)

        if batch:
            for overlay in process_batch(
                model,
                batch,
                imgsz,
                device,
                line_detector,
                parking_dot_detector,
            ):
                writer.write(overlay)
                written += 1
    finally:
        capture.release()
        writer.release()
        progress.set_postfix(written=written)
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/parking"))
    parser.add_argument("--output", type=Path, default=Path("result/parking/yolo_sem_class_video"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960, help="Frame width. 0 keeps the source size.")
    parser.add_argument("--height", type=int, default=540, help="Frame height. 0 keeps the source size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0")
    if args.batch <= 0:
        raise SystemExit("--batch must be greater than 0")
    if (args.width == 0) != (args.height == 0):
        raise SystemExit("--width and --height must both be 0 or both be positive")
    if args.width < 0 or args.height < 0:
        raise SystemExit("--width and --height cannot be negative")

    uses_onnx = args.backend == "onnx" or (
        args.backend == "auto" and args.weights.suffix.lower() == ".onnx"
    )
    if uses_onnx and args.batch != 1:
        print(
            f"ONNX inference uses batch size 1 (requested --batch {args.batch}).",
            flush=True,
        )
        args.batch = 1

    model = load_semantic_model(args.weights, args.backend)
    sources = sorted(args.input.glob("*.mp4"))
    if not sources:
        raise SystemExit(f"No mp4 files found in {args.input}")

    for source in sources:
        render_video(
            model=model,
            source=source,
            destination=args.output / source.name,
            output_fps=args.fps,
            imgsz=args.imgsz,
            device=args.device,
            batch_size=args.batch,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
