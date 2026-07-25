from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils.perspective import (
    apply_new_perspective,
    calibration_errors,
    make_new_perspective_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp extracted parking frames into the map.svg coordinate system."
    )
    parser.add_argument("--input", type=Path, default=Path("dataset/parking/frames"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/parking/perspective"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_dirs = sorted(path for path in args.input.iterdir() if path.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {args.input}")

    total = 0
    reported_error = False
    for frame_dir in frame_dirs:
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            print(f"{frame_dir.name}: no frames, skipped")
            continue

        output_dir = args.output / frame_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")

            config = make_new_perspective_config(frame.shape)
            if not reported_error:
                errors = calibration_errors(config)
                print(
                    "Calibration reprojection error: "
                    f"mean={errors.mean():.2f}px, max={errors.max():.2f}px"
                )
                reported_error = True

            warped = apply_new_perspective(frame, config)
            output_path = output_dir / frame_path.name
            if not cv2.imwrite(str(output_path), warped):
                raise RuntimeError(f"Could not write result: {output_path}")
            count += 1

        total += count
        print(f"{frame_dir.name}: {count} frames")

    print(f"Total: {total} frames")


if __name__ == "__main__":
    main()
