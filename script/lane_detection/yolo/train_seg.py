#!/usr/bin/env python3
"""Train a YOLO segmentation model for road/lane masks."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/lane_detection/yolo_lane_seg/data.yaml"))
    parser.add_argument("--model", default="weights/yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/segment/yolo_lane_seg"))
    parser.add_argument("--name", default="train_cpu_640_yolo26n")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in the active environment. "
            "Install it in the overdrive conda env before training."
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        task="segment",
        patience=30,
        close_mosaic=20,
        degrees=2.0,
        translate=0.04,
        scale=0.25,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,
        plots=True,
    )


if __name__ == "__main__":
    main()
