#!/usr/bin/env python3
"""OVERDRIVE autonomous driving entry point.

현재는 기존 오른쪽 차선 추종 프로그램을 그대로 실행한다.
추후 정지선, 신호등, 차선 변경, 라이다 기능을 이 파일에서 통합한다.
"""

from __future__ import annotations

import sys

from control.realtime_drive import (
    main as run_right_lane_following,
)


def main() -> None:
    """Run the existing right-lane-following program."""
    print("=" * 60)
    print("OVERDRIVE")
    print("Mode: right-lane following")
    print("=" * 60)

    try:
        run_right_lane_following()

    except KeyboardInterrupt:
        print("\nOVERDRIVE stopped by user.", flush=True)

    except Exception as exc:
        print(
            f"\nOVERDRIVE failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()