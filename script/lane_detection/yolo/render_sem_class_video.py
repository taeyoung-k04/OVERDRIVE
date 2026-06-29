#!/usr/bin/env python3
"""Render videos using the classified lane semantic model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_sem_class import make_class_overlay, semantic_to_class_map


def process_batch(model, frames: list[np.ndarray], imgsz: int, device: str) -> list[np.ndarray]:
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
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or output_fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {destination}")

    frame_index = 0
    written = 0
    next_time = 0.0
    frame_step_time = 1.0 / output_fps
    batch: list[np.ndarray] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        time_seconds = frame_index / source_fps
        frame_index += 1
        if time_seconds + 1e-9 < next_time:
            continue

        batch.append(frame)
        next_time += frame_step_time
        if len(batch) >= batch_size:
            for overlay in process_batch(model, batch, imgsz, device):
                writer.write(overlay)
                written += 1
            batch.clear()
            print(f"{source.name}: read {frame_index}/{total_frames}, wrote {written}", flush=True)

    if batch:
        for overlay in process_batch(model, batch, imgsz, device):
            writer.write(overlay)
            written += 1

    capture.release()
    writer.release()
    print(f"Saved {destination} ({written} frames @ {output_fps:g} FPS)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection"))
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_sem_class_video"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in the active environment. "
            "Install it in the overdrive conda env before rendering videos."
        ) from exc

    model = YOLO(str(args.weights), task="semantic")
    sources = sorted(args.input.glob("*.mp4"))
    if not sources:
        raise SystemExit(f"No mp4 files found in {args.input}")

    for source in sources:
        render_video(model, source, args.output / source.name, args.fps, args.imgsz, args.device, args.batch)


if __name__ == "__main__":
    main()
