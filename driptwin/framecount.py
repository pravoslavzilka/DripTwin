"""Read the drop counter that vision.DropDetector.annotate draws on each frame
("drops N", top left), from a running server's /frame.jpg. Lets a second
program follow the count while that server keeps the camera."""

from __future__ import annotations

import threading
import time
import urllib.request

import cv2
import numpy as np

FONT, SCALE, THICK, ORG = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, (8, 20)
H = 28                                  # rows of the label strip


def _render(text: str, width: int) -> np.ndarray:
    img = np.zeros((H, width), np.uint8)
    cv2.putText(img, text, ORG, FONT, SCALE, 255, THICK)
    return img


def _mask(frame: np.ndarray) -> np.ndarray:
    strip = frame[:H]
    return (strip.min(axis=2) > 170).astype(np.uint8) * 255   # white text only


def read_count(frame: np.ndarray) -> int | None:
    """Greedy digit-by-digit match of 'drops N' against the frame's label."""
    m = _mask(frame).astype(np.float32)
    w = m.shape[1]
    digits = ""
    for _ in range(9):
        scores = {}
        for d in "0123456789":
            t = "drops " + digits + d
            x1 = cv2.getTextSize(t, FONT, SCALE, THICK)[0][0] + ORG[0] + 2
            x0 = cv2.getTextSize("drops " + digits, FONT, SCALE, THICK)[0][0] + ORG[0] - 1
            ref = _render(t, w).astype(np.float32)
            a, b = ref[:, x0:x1], m[:, x0:x1]
            scores[d] = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
        best = max(scores, key=scores.get)
        if scores[best] < 0.6:
            break
        digits += best
    return int(digits) if digits else None


class FrameDrops(threading.Thread):
    """Polls /frame.jpg and calls on_drop(t) once per counter increment."""

    def __init__(self, on_drop, url: str = "http://127.0.0.1:8000/frame.jpg", hz: float = 12.0,
                 min_interval_s: float = 0.0):
        """min_interval_s drops counts that follow the previous one too closely
        (the server's ripple after a splash, when its refractory is short)."""
        super().__init__(daemon=True)
        self.on_drop, self.url, self.period = on_drop, url, 1.0 / hz
        self.min_interval_s = min_interval_s
        self.count: int | None = None
        self.last_t = -1e9

    def run(self) -> None:
        while True:
            try:
                b = urllib.request.urlopen(self.url, timeout=2).read()
                n = read_count(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
            except Exception:
                n = None
            if n is not None:
                if self.count is not None and 0 < n - self.count <= 5:   # ignore misreads
                    t = time.monotonic()
                    if t - self.last_t >= self.min_interval_s:
                        for _ in range(1 if self.min_interval_s else n - self.count):
                            self.on_drop(t)
                        self.last_t = t
                if self.count is None or 0 < n - self.count <= 5:
                    self.count = n
            time.sleep(self.period)
