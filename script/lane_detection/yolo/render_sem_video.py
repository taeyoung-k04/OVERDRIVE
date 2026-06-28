#!/usr/bin/env python3
"""Render lane-recognition videos using the trained YOLO semantic model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rules import make_overlay
from infer_sem import LANE_ID, ROAD_ID


def semantic_to_masks(class_map: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if class_map.shape != shape:
        class_map = cv2.resize(class_map, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    road = (class_map == ROAD_ID).astype(np.uint8) * 255
    lane = (class_map == LANE_ID).astype(np.uint8) * 255
    return road, lane


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
        semantic = result.semantic_mask
        if semantic is None:
            road = np.zeros(frame.shape[:2], dtype=np.uint8)
            lane = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            class_map = semantic.data.cpu().numpy().astype(np.uint8)
            road, lane = semantic_to_masks(class_map, frame.shape[:2])
        overlays.append(make_overlay(frame, lane, road))
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
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_sem_video"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/semantic/runs/yolo_lane_sem/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
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
        destination = args.output / source.name
        render_video(model, source, destination, args.fps, args.imgsz, args.device, args.batch)


if __name__ == "__main__":
    main()
