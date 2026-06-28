#!/usr/bin/env python3
"""Train a YOLO semantic segmentation model for road/lane pixels."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/lane_detection/yolo_lane_sem/data.yaml"))
    parser.add_argument(
        "--model",
        default="weights/yolo26n-sem-ade20k.pt",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/semantic/runs/yolo_lane_sem"))
    parser.add_argument("--name", default="train_cpu_640_yolo26n_ade20k")
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

    model = YOLO(str(args.model), task="semantic")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        task="semantic",
        patience=25,
        degrees=2.0,
        translate=0.04,
        scale=0.20,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,
        copy_paste=0.0,
        mosaic=0.0,
        mixup=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
