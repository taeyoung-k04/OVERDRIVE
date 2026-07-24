#!/usr/bin/env python3
"""Entry point for autonomous driving."""

from __future__ import annotations

from control.realtime_drive import (
    main as run_right_lane_following,
)


def main() -> None:
    run_right_lane_following()


if __name__ == "__main__":
    main()
