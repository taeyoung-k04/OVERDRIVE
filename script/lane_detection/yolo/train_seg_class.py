#!/usr/bin/env python3
"""Train YOLO segmentation for classified OVERDRIVE road objects."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/lane_detection/yolo_lane_seg_class/data.yaml"))
    parser.add_argument("--model", type=Path, default=Path("weights/yolo26n-seg.pt"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/segment/yolo_lane_seg_class"))
    parser.add_argument("--name", default="train_cpu_960_yolo26n")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
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

    model = YOLO(str(args.model.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        task="segment",
        workers=args.workers,
        patience=args.patience,
        exist_ok=True,
        close_mosaic=20,
        degrees=2.0,
        translate=0.04,
        scale=0.20,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,
        copy_paste=0.0,
        mixup=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
