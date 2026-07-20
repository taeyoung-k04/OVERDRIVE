#!/usr/bin/env python3
"""Camera input utilities for OVERDRIVE."""

from __future__ import annotations

import sys
import threading
from typing import Optional

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    errors: list[str] = []
    for backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            capture.set(cv2.CAP_PROP_FPS, fps)

        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                return capture
            errors.append(f"backend={backend}: opened but frame read failed")
        else:
            errors.append(f"backend={backend}: open failed")
        capture.release()

    raise RuntimeError(
        f"Could not open camera index {camera_index}. " + "; ".join(errors)
    )


class LatestFrameReader:
    """Continuously drain the camera and expose only the most recent frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self.condition = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.stopped = False
        self.failed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            with self.condition:
                if self.stopped:
                    return

            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.failed = True
                    self.stopped = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.condition.notify_all()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self.condition:
            if self.frame is None and not self.stopped:
                self.condition.wait(timeout=2.0)
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)


def normalize_frame_size(
    frame: np.ndarray,
    width: int,
    height: int,
    force_size: bool,
) -> np.ndarray:
    if not force_size or width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


