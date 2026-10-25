"""OpenCV drop counter for the drip chamber.

Counts splashes. One region of interest, the splash box, sits where each drop
lands in the fluid at the bottom of the chamber. A landing throws up a burst of
light and ripples that is far bigger than the drop itself and always in the same
place, so it is the most reliable thing to count (on a real recording: 75/75
drops from the splash vs 51/75 from the falling drop).

The signal is reference-free: mean absolute frame-to-frame change inside the
box. The threshold adapts to the scene noise (rolling median + k * MAD), so
lighting changes don't need retuning. A refractory window stops one drop's
splash and ripples being counted twice. A change outside the box as big as the
one inside (a hand, a bump, a light) never counts.

With no saved box, the detector watches the first ~10 s and puts the box on the
spot that changes most, which is the splash. Run standalone to check or adjust:
    python -m driptwin.vision --camera 0 --calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROI_FILE = Path(__file__).resolve().parent.parent / "rois.json"


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y:self.y + self.h, self.x:self.x + self.w]


@dataclass
class ROIConfig:
    splash: ROI

    def save(self, path: Path = ROI_FILE) -> None:
        path.write_text(json.dumps({"splash": asdict(self.splash)}))

    @classmethod
    def load(cls, path: Path = ROI_FILE) -> "ROIConfig | None":
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        roi = ROI(**(saved.get("splash") or saved["orifice"]))   # "orifice": older files
        return cls(roi) if roi.w > 0 and roi.h > 0 else None   # empty box: find it again


LOCATE_SCALE = 4        # splash search runs on quarter-size difference images


def locate_splash(diffs: np.ndarray, scale: int = LOCATE_SCALE, frac: float = 0.5) -> ROI | None:
    """Box around the spot that changes most over a few seconds of difference
    images: per pixel, the 99th percentile of change; then the biggest blob above
    half the peak. Drops land every second or so, so the splash dominates; a
    one-off movement only touches a few frames and stays under the 99th percentile."""
    m = np.percentile(diffs, 99, axis=0)
    if m.max() < 8:                    # nothing moving: no drops yet
        return None
    mask = (m > frac * m.max()).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) * scale for v in stats[i, :4])
    pad = 2 * scale
    return ROI(max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad)


class SpikeDetector:
    """Adaptive threshold on a 1-D signal: fires on a rising edge above
    max(median + k*MAD, ratio * median, 0.4 * largest recent drop spike),
    then waits `refractory_s`. Learns the scene for `warmup` frames first.

    Tuned on a real recording with hand-marked drops (driptwin.groundtruth):
    - Only quiet frames feed the noise estimate. A real ripple lasts several
      frames and drops come about every second, so letting drop frames in (or
      using a high percentile, as an earlier version did) lifts the threshold
      above the drops and nothing is ever counted again.
    - ratio * median keeps the small, frequent changes of a growing pendant drop
      from counting.
    - The drop scale forgets after `forget_s` without a drop, so one odd spike
      can't blind the detector for good.
    - `disturbed` frames (the whole picture changed: a hand, a bump, a light)
      never count.
    """

    def __init__(self, k: float = 6.0, history: int = 150, refractory_s: float = 0.3,
                 floor: float = 1.5, ratio: float = 2.0, warmup: int = 90, forget_s: float = 8.0):
        self.k, self.refractory_s, self.floor, self.ratio = k, refractory_s, floor, ratio
        self.warmup, self.forget_s = warmup, forget_s
        self.hist: deque[float] = deque(maxlen=history)
        self.spikes: deque[float] = deque(maxlen=9)    # heights of recent drop spikes
        self.last_fire = -1e9
        self.above = False
        self.threshold = float("inf")

    def update(self, x: float, t: float, disturbed: bool = False) -> bool:
        if len(self.hist) < self.warmup:       # learn the scene before counting
            self.hist.append(x)
            return False
        if self.spikes and t - self.last_fire > self.forget_s:
            self.spikes.clear()                # long gap: relearn the drop size
        h = np.fromiter(self.hist, float)
        # Noise level from the quietest frames only. At fast drips splash frames are
        # a big share of all frames; once one slipped into the history it lifted the
        # median and spread, the threshold rose over the next spike, that spike was
        # learned as noise too, and counting stopped for good.
        p5, p30 = np.percentile(h, (5, 30))
        sigma = (p30 - p5) / 1.12                     # gaussian: p30 - p5 = 1.12 sigma
        base = p30 + 0.52 * sigma                     # ~ the noise median
        self.threshold = max(base + max(self.k * sigma, self.floor), self.ratio * base,
                             0.4 * max(self.spikes, default=0.0))
        fire = False
        if x > self.threshold:
            if not self.above and t - self.last_fire > self.refractory_s:
                self.last_fire = t
                if not disturbed:
                    fire = True
                    self.spikes.append(x)
            self.above = True
        else:
            self.above = False
            self.hist.append(x)                # noise estimate from quiet frames only
        return fire


class DropDetector:
    """rois=None: find the splash box automatically from the first `locate_frames`."""

    def __init__(self, rois: ROIConfig | None, refractory_s: float = 0.3,
                 locate_frames: int = 300):
        self.rois = rois
        self.refractory_s = refractory_s
        self.locate_frames = locate_frames
        self._seen: list[np.ndarray] = []              # small diffs while locating
        self.det_splash = SpikeDetector()
        self.scene: deque[float] = deque(maxlen=300)    # change outside the box, last ~10 s
        self._busy_run = 0                              # frames in a row the outside was busy
        self.prev: np.ndarray | None = None
        self.last_drop = -1e9
        self.count = 0
        self.trace: deque[float] = deque(maxlen=200)   # for the preview plot

    @property
    def locating(self) -> bool:
        return self.rois is None

    @staticmethod
    def _prep(frame: np.ndarray) -> np.ndarray:
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0)

    def process(self, frame: np.ndarray, t: float) -> bool:
        g = self._prep(frame)
        if self.prev is None or self.prev.shape != g.shape:
            self.prev = g
            return False
        diff = cv2.absdiff(g, self.prev)
        self.prev = g
        if self.rois is None:
            self._locate(diff)
            return False
        r = self.rois.splash
        box = r.crop(diff)
        s_box = float(box.mean())
        area_all, area_box = diff.size, box.size
        s_out = (float(diff.mean()) * area_all - s_box * area_box) / max(area_all - area_box, 1)
        # A splash flashes the backlight across the chamber: about a fifth of the
        # picture outside the box changes too, for 2-3 frames, less than the box
        # does. A hand, a bump or a light changes most of the picture, or more
        # outside than in the box, or keeps changing it for many frames.
        changed_out = (np.count_nonzero(diff > 20) - np.count_nonzero(box > 20)) / max(area_all - area_box, 1)
        busy = len(self.scene) > 30 and s_out > 4.0 * float(np.median(self.scene)) + 0.5
        disturbed = busy and (changed_out > 0.5 or s_out > 1.5 * s_box or self._busy_run >= 4)
        self._busy_run = self._busy_run + 1 if busy else 0
        self.scene.append(s_out)
        fired = self.det_splash.update(s_box, t, disturbed)
        self.trace.append(s_box)
        if fired and t - self.last_drop > self.refractory_s:
            self.last_drop = t
            self.count += 1
            return True
        return False

    def _locate(self, diff: np.ndarray) -> None:
        self._seen.append(cv2.resize(diff, None, fx=1 / LOCATE_SCALE, fy=1 / LOCATE_SCALE,
                                     interpolation=cv2.INTER_AREA))
        if len(self._seen) >= self.locate_frames:
            roi = locate_splash(np.stack(self._seen))
            self._seen = []                            # nothing moving: look again
            if roi:
                self.rois = ROIConfig(roi)
                print(f"[vision] splash box found: {asdict(roi)}")

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.rois is None:
            left = (self.locate_frames - len(self._seen)) / 30
            cv2.putText(img, f"finding the splash... {left:.0f} s", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return img
        flash = time.monotonic() - self.last_drop < 0.15
        roi = self.rois.splash
        cv2.rectangle(img, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h),
                      (0, 255, 0) if flash else (0, 200, 255), 2)
        cv2.putText(img, f"drops {self.count}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        # sparkline of the splash signal vs its adaptive threshold
        h, w = img.shape[:2]
        if len(self.trace) > 2:
            vals = np.array(self.trace)
            top = max(vals.max(), self.det_splash.threshold if np.isfinite(
                self.det_splash.threshold) else 1, 1e-3)
            pts = [(int(i * w / len(vals)), int(h - 4 - v / top * 40)) for i, v in enumerate(vals)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 200, 255), 1)
            if np.isfinite(self.det_splash.threshold):
                ty = int(h - 4 - self.det_splash.threshold / top * 40)
                cv2.line(img, (0, ty), (w, ty), (80, 80, 80), 1)
        return img


def default_rois(w: int, h: int) -> ROIConfig:
    """Fallback box: lower middle, where the fluid surface usually is."""
    return ROIConfig(ROI(int(w * 0.3), int(h * 0.55), int(w * 0.4), int(h * 0.3)))


DISPLAY_W = 960      # preview windows wider than this lag on a laptop screen


def _measure_fps(cap: cv2.VideoCapture, frames: int = 12) -> float:
    cap.read()
    t = time.monotonic()
    got = sum(cap.read()[0] for _ in range(frames))
    return got / max(time.monotonic() - t, 1e-3)


def open_camera(camera: int | str) -> cv2.VideoCapture:
    """Webcams are opened at 1280x720/30 fps: plenty for a drop, and calibration
    and counting always see the same pixel grid, so the saved box lines up.

    On Windows, DirectShow: the default backend can take a minute to open a
    camera. Some webcams only send uncompressed video there, which manages
    ~5 fps at 720p; a falling drop needs ~30, so those drop to 640x480."""
    if isinstance(camera, str):
        return cv2.VideoCapture(camera)
    win = sys.platform == "win32"
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if win else cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if win and cap.isOpened() and _measure_fps(cap) < 20:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[vision] camera {camera} is slow at 720p; using 640x480 for 30 fps")
    return cap


def fit(img: np.ndarray, width: int = DISPLAY_W) -> tuple[np.ndarray, float]:
    """Downscale for display only; returns the image and the scale used."""
    s = min(1.0, width / img.shape[1])
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img), s


class CameraCounter(threading.Thread):
    """Background thread: reads the webcam, calls on_drop(t) per drop, and keeps
    the latest annotated JPEG for the dashboard."""

    def __init__(self, on_drop: Callable[[float], None], camera: int | str = 0,
                 rois: ROIConfig | None = None):
        super().__init__(daemon=True)
        self.on_drop = on_drop
        self.camera = camera
        self.rois = rois
        self.jpeg: bytes | None = None
        self.fps = 0.0
        self.alive = False
        self.error: str | None = None
        self._halt = threading.Event()

    def run(self) -> None:
        # Keep trying: on Windows a camera another program just released can refuse
        # to open for a moment, and at a demo it may be plugged in late.
        while True:
            cap = open_camera(self.camera)
            ok, frame = cap.read()
            if ok or isinstance(self.camera, str) or self._halt.wait(2.0):
                break
            cap.release()
            if self.error is None:
                self.error = f"camera {self.camera} not available"
                print(f"[vision] {self.error}; retrying every 2 s")
        if not ok:
            self.error = f"camera {self.camera} not available"
            print(f"[vision] {self.error}")
            return
        self.error = None
        det = DropDetector(self.rois or ROIConfig.load())      # None: find the splash
        self.alive = True
        # A video file would otherwise be read as fast as the CPU allows, which
        # squashes time and inflates the rate: play it at its own fps, looping.
        is_file = isinstance(self.camera, str)
        period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
        n, t_fps, t_next = 0, time.monotonic(), time.monotonic()
        while not self._halt.is_set():
            if period:
                t_next += period
                time.sleep(max(0.0, t_next - time.monotonic()))
            ok, frame = cap.read()
            if not ok and is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                self.alive = False
                time.sleep(0.05)
                continue
            self.alive = True
            t = time.monotonic()
            if det.process(frame, t):
                self.on_drop(t)
            n += 1
            if n % 2 == 0:      # dashboard feed: 15 fps of small JPEGs keeps the browser in step
                self.jpeg = cv2.imencode(".jpg", fit(det.annotate(frame), 640)[0],
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            if t - t_fps > 1.0:
                self.fps, n, t_fps = n / (t - t_fps), 0, t
        cap.release()

    def stop(self) -> None:
        self._halt.set()


def calibrate(camera: int | str) -> None:
    """Live box setup: drag a box on the moving video and the detector runs on it
    straight away, so the box can be tuned against real drops before saving."""
    cap = open_camera(camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"camera {camera} not available")
    saved = ROIConfig.load()
    det = DropDetector(saved)                   # no saved box: find the splash
    win = "calibrate: drag a box | a = auto | s = save | r = reset | q = quit"
    drag: dict = {}
    scale = fit(frame)[1]

    def on_mouse(event, x, y, flags, _):
        nonlocal det
        x, y = round(x / scale), round(y / scale)          # display -> camera pixels
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(start=(x, y), end=(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and "start" in drag:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and "start" in drag:
            (x0, y0), (x1, y1) = drag.pop("start"), (x, y)
            drag.pop("end", None)
            if abs(x1 - x0) > 4 and abs(y1 - y0) > 4:
                det = DropDetector(ROIConfig(ROI(min(x0, x1), min(y0, y1),
                                                 abs(x1 - x0), abs(y1 - y0))))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Put the box over the splash, where drops land in the fluid. Press a to find it\n"
          "automatically (watches ~10 s of drops), or drag a box on the live video.\n"
          "Each drop should flash the box green once (after ~3 s warm-up).\n"
          "s = save, r = reset to the saved box, q = quit without saving.")
    # video files: wait one frame period so drops arrive in real time
    wait_ms = max(1, int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30))) if isinstance(camera, str) else 1
    while ok:
        det.process(frame, time.monotonic())
        img = det.annotate(frame)
        if "start" in drag:
            cv2.rectangle(img, drag["start"], drag["end"], (255, 255, 255), 1)
        if not det.locating and len(det.det_splash.hist) < det.det_splash.warmup:
            cv2.putText(img, "learning the scene...", (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)
        cv2.imshow(win, fit(img)[0])
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("s") and not det.locating:
            det.rois.save()
            print(f"saved {ROI_FILE}: {asdict(det.rois.splash)}")
            break
        if key == ord("a"):
            det = DropDetector(None)
        if key == ord("r"):
            det = DropDetector(saved)
        if key == ord("q"):
            print("quit, nothing saved")
            break
        ok, nxt = cap.read()
        frame = nxt if ok else frame
    cap.release()
    cv2.destroyAllWindows()


def preview(camera: int | str) -> None:
    cap = open_camera(camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"camera {camera} not available")
    det = DropDetector(ROIConfig.load())       # no saved box: find the splash
    drops: deque[float] = deque(maxlen=10)
    # video files: wait one frame period so playback (and the rate) is real time
    wait_ms = max(1, int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30))) if isinstance(camera, str) else 1
    while ok:
        t = time.monotonic()
        if det.process(frame, t):
            drops.append(t)
        img = det.annotate(frame)
        if len(drops) >= 3:
            rate = 60.0 * (len(drops) - 1) / (drops[-1] - drops[0])
            cv2.putText(img, f"{rate:.1f} gtt/min", (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        cv2.imshow("DripTwin vision (q to quit)", fit(img)[0])
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
        ok, frame = cap.read()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    cam = int(a.camera) if a.camera.isdigit() else a.camera
    calibrate(cam) if a.calibrate else preview(cam)
