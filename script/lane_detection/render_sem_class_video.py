#!/usr/bin/env python3
"""Render videos using the lane, car, and traffic-light semantic model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from infer_sem_class import (
    DEFAULT_WEIGHTS,
    load_semantic_model,
    make_class_overlay,
    semantic_to_class_map,
)
from utils.postprocess import (
    add_postprocess_args,
    postprocess_class_map,
)


def process_batch(
    model,
    frames: list[np.ndarray],
    imgsz: int,
    device: str,
    postprocess: bool,
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
        class_map = semantic_to_class_map(result.semantic_mask, frame.shape[:2])
        if postprocess:
            class_map = postprocess_class_map(class_map)
        overlays.append(make_class_overlay(frame, class_map))
    return overlays


def render_video(
    model,
    source: Path,
    destination: Path,
    output_fps: float,
    imgsz: int,
    device: str,
    batch_size: int,
    postprocess: bool,
    args: argparse.Namespace,
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or output_fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    force_size = args.width > 0 and args.height > 0
    frame_width = args.width if force_size else source_width
    frame_height = args.height if force_size else source_height

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

    progress = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=source.stem,
        unit="frame",
    )
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
            for overlay in process_batch(model, batch, imgsz, device, postprocess):
                writer.write(overlay)
                written += 1
            batch.clear()
            progress.set_postfix(written=written)

    if batch:
        for overlay in process_batch(model, batch, imgsz, device, postprocess):
            writer.write(overlay)
            written += 1

    capture.release()
    writer.release()
    progress.set_postfix(written=written)
    progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection"))
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_sem_class_video"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to .pt or .onnx weights. With --backend onnx, a .pt suffix is replaced with .onnx.",
    )
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960, help="Frame width. 0 keeps the source size.")
    parser.add_argument("--height", type=int, default=540, help="Frame height. 0 keeps the source size.")
    add_postprocess_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    uses_onnx = args.backend == "onnx" or (args.backend == "auto" and args.weights.suffix.lower() == ".onnx")
    if uses_onnx and args.batch != 1:
        print(
            f"ONNX inference uses batch size 1 (requested --batch {args.batch}). "
            "Export the model with dynamic batching to use larger batches.",
            flush=True,
        )
        args.batch = 1

    model = load_semantic_model(args.weights, args.backend)
    sources = sorted(args.input.glob("*.mp4"))
    if not sources:
        raise SystemExit(f"No mp4 files found in {args.input}")

    for source in sources:
        render_video(
            model,
            source,
            args.output / source.name,
            args.fps,
            args.imgsz,
            args.device,
            args.batch,
            args.postprocess,
            args,
        )


if __name__ == "__main__":
    main()
