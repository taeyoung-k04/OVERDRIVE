#!/usr/bin/env python3
"""Render videos with classified YOLO segmentation alpha overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_seg_class import make_alpha_overlay, result_to_masks


def process_batch(model, frames: list[np.ndarray], imgsz: int, conf: float, device: str, alpha: float) -> list[np.ndarray]:
    results = model.predict(source=frames, imgsz=imgsz, conf=conf, device=device, verbose=False)
    overlays: list[np.ndarray] = []
    for frame, result in zip(frames, results):
        names = getattr(result, "names", None) or getattr(model, "names", {})
        overlays.append(make_alpha_overlay(frame, result_to_masks(result, frame.shape[:2], conf), names, alpha))
    return overlays


def render_video(
    model,
    source: Path,
    destination: Path,
    output_fps: float,
    imgsz: int,
    conf: float,
    device: str,
    alpha: float,
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
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
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
            for overlay in process_batch(model, batch, imgsz, conf, device, alpha):
                writer.write(overlay)
                written += 1
            batch.clear()
            print(f"{source.name}: read {frame_index}/{total_frames}, wrote {written}", flush=True)

    if batch:
        for overlay in process_batch(model, batch, imgsz, conf, device, alpha):
            writer.write(overlay)
            written += 1

    capture.release()
    writer.release()
    print(f"Saved {destination} ({written} frames @ {output_fps:g} FPS)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection"))
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_seg_class_video"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/yolo_lane_seg_class/train_cpu_960_yolo26n/weights/best.pt"),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.35)
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

    model = YOLO(str(args.weights.resolve()))
    sources = sorted(args.input.glob("*.mp4"))
    if not sources:
        raise SystemExit(f"No mp4 files found in {args.input}")

    for source in sources:
        render_video(model, source, args.output / source.name, args.fps, args.imgsz, args.conf, args.device, args.alpha, args.batch)


if __name__ == "__main__":
    main()
