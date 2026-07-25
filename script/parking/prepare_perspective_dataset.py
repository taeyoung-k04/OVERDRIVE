from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils.perspective import apply_perspective, make_perspective_config


def apply_perspective_to_frames(
    frame_paths: list[Path],
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    perspective_args = argparse.Namespace(perspective=True)

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        config = make_perspective_config(perspective_args, frame.shape)
        warped = apply_perspective(frame, config)
        output_path = output_dir / frame_path.name
        if not cv2.imwrite(str(output_path), warped):
            raise RuntimeError(f"Could not write perspective result: {output_path}")

    return len(frame_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply perspective warp to extracted parking frames."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/parking/frames"),
        help="Folder containing one subdirectory per video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/parking/perspective"),
        help="Folder in which perspective results are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_dirs = sorted(path for path in args.input.iterdir() if path.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {args.input}")

    total = 0
    for frame_dir in frame_dirs:
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            print(f"{frame_dir.name}: no frames, skipped")
            continue

        result_dir = args.output / frame_dir.name
        count = apply_perspective_to_frames(frame_paths, result_dir)
        total += count
        print(f"{frame_dir.name}: {count} frames")

    print(f"Total: {total} frames")


if __name__ == "__main__":
    main()
