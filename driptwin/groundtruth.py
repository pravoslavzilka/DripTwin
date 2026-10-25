"""Measure real drop-counting accuracy against a human count.

    python -m driptwin.groundtruth record --camera 0 --seconds 120 --name run1
        Live view with the saved box. Press SPACE every time you see a drop fall,
        q to stop early. Saves recordings/run1.mp4 and recordings/run1.json.

    python -m driptwin.groundtruth eval recordings/run1.mp4 [--box x,y,w,h]
        Runs the detector on the video, matches its drops to your SPACE presses,
        prints recall / precision / count error, and saves recordings/run1_signal.png.

A press usually comes 0.2-0.8 s after the drop detaches (reaction time), so a
detection counts as correct if a press follows it within MATCH_AFTER_S, or
precedes it by at most MATCH_BEFORE_S.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .vision import DropDetector, ROI, ROIConfig, default_rois, fit, open_camera

OUT = Path(__file__).resolve().parent.parent / "recordings"
MATCH_BEFORE_S, MATCH_AFTER_S = 0.3, 1.0


def record(camera: int | str, seconds: float, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    cap = open_camera(camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"camera {camera} not available")
    rois = ROIConfig.load() or default_rois(frame.shape[1], frame.shape[0])
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    times: list[float] = []      # capture time of every frame, so eval replays real timing
    taps: list[int] = []         # frame index at each SPACE press
    win = "record: SPACE or click = a drop fell | q = stop"
    cv2.namedWindow(win)
    # A click in the window marks a drop too: it works even when the window has no keyboard focus.
    cv2.setMouseCallback(win, lambda ev, *_: taps.append(len(times) - 1)
                         if ev == cv2.EVENT_LBUTTONDOWN and times else None)
    cv2.imshow(win, fit(frame)[0])
    cv2.waitKey(1)
    _focus(win)
    print(f"Recording {seconds:.0f} s. Press SPACE or click the window each time a drop falls. "
          "q stops early.")
    t0 = time.monotonic()
    while ok and time.monotonic() - t0 < seconds:
        times.append(time.monotonic() - t0)
        writer.write(frame)
        r = rois.splash
        view = frame.copy()
        cv2.rectangle(view, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 200, 255), 2)
        cv2.putText(view, f"{times[-1]:5.1f} s   drops you marked: {len(taps)}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, fit(view)[0])
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            taps.append(len(times) - 1)
        elif key == ord("q"):
            break
        ok, frame = cap.read()
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    meta = {"frame_times": times, "taps": taps, "box": vars(rois.splash)}
    (OUT / f"{name}.json").write_text(json.dumps(meta))
    print(f"saved {OUT / name}.mp4 and .json: {len(times)} frames "
          f"({len(times) / times[-1]:.1f} fps), {len(taps)} drops marked")


def _focus(win: str) -> None:
    """Bring the window to the front with keyboard focus. On macOS a window opened
    from a background process otherwise stays behind and never sees SPACE."""
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", 'tell application "System Events" to set frontmost of '
                        f'(first process whose unix id is {os.getpid()}) to true'],
                       capture_output=True, timeout=5)


def match(detected: list[float], truth: list[float]) -> tuple[int, list[float], list[float]]:
    """Greedy one-to-one matching in time order. Returns (matched, missed truth, false detections)."""
    truth_left, false = list(truth), []
    matched = 0
    for d in detected:
        hit = next((t for t in truth_left if -MATCH_BEFORE_S <= t - d <= MATCH_AFTER_S), None)
        if hit is None:
            false.append(d)
        else:
            truth_left.remove(hit)
            matched += 1
    return matched, truth_left, false


def evaluate(video: str, box: str | None) -> None:
    path = Path(video)
    meta = json.loads(path.with_suffix(".json").read_text())
    times, taps = meta["frame_times"], meta["taps"]
    roi = ROI(*map(int, box.split(","))) if box else ROI(**meta["box"])
    det = DropDetector(ROIConfig(roi))
    sd = det.det_splash
    cap = cv2.VideoCapture(str(path))
    detected, signal, thresh = [], [], []
    for t in times:
        ok, frame = cap.read()
        if not ok:
            break
        if det.process(frame, t):
            detected.append(t)
        signal.append(det.trace[-1] if det.trace else 0.0)
        thresh.append(sd.threshold if np.isfinite(sd.threshold) else 0.0)
    warm = times[min(sd.warmup, len(times) - 1)]          # the detector counts nothing before this
    truth = [times[i] for i in taps if times[i] >= warm]
    detected = [d for d in detected if d >= warm]
    matched, missed, false = match(detected, truth)
    n = len(truth)
    print(f"box {vars(roi)}   (first {warm:.1f} s skipped: detector warm-up)")
    print(f"drops you marked : {n}")
    print(f"drops detected   : {len(detected)}")
    print(f"  correct        : {matched}")
    print(f"  missed         : {len(missed)}" + (f"  at {', '.join(f'{m:.1f}' for m in missed[:15])} s" if missed else ""))
    print(f"  false          : {len(false)}" + (f"  at {', '.join(f'{f:.1f}' for f in false[:15])} s" if false else ""))
    if n:
        print(f"recall {matched / n:.0%}   precision {matched / max(1, len(detected)):.0%}   "
              f"count error {(len(detected) - n) / n:+.0%}")
    png = path.with_name(path.stem + "_signal.png")
    cv2.imwrite(str(png), plot(times[:len(signal)], signal, thresh, detected, truth, false))
    print(f"signal plot: {png}  (grey = signal, red = threshold, green = detected, blue = your SPACE)")


def plot(times, signal, thresh, detected, truth, false) -> np.ndarray:
    W, H, pad = 1600, 420, 30
    img = np.full((H, W, 3), 255, np.uint8)
    top = max(max(signal, default=1), max(thresh, default=1), 1e-3) * 1.1
    span = max(times[-1], 1e-3)
    x = lambda t: int(pad + (W - 2 * pad) * t / span)
    y = lambda v: int(H - pad - (H - 2 * pad) * min(v, top) / top)
    for t in truth:
        cv2.line(img, (x(t), pad), (x(t), H - pad), (255, 140, 0), 1)                 # blue: you
    for t in detected:
        cv2.line(img, (x(t), H - pad), (x(t), H - pad - 25), (0, 160, 0) if t not in false else (0, 0, 255), 3)
    cv2.polylines(img, [np.array([(x(t), y(v)) for t, v in zip(times, signal)], np.int32)], False, (90, 90, 90), 1)
    cv2.polylines(img, [np.array([(x(t), y(v)) for t, v in zip(times, thresh)], np.int32)], False, (0, 0, 220), 1)
    for s in range(0, int(span) + 1, 10):
        cv2.putText(img, f"{s}s", (x(s), H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--camera", default="0")
    r.add_argument("--seconds", type=float, default=120)
    r.add_argument("--name", default=time.strftime("run-%H%M%S"))
    e = sub.add_parser("eval")
    e.add_argument("video")
    e.add_argument("--box", help="try a different box: x,y,w,h")
    a = ap.parse_args()
    if a.cmd == "record":
        record(int(a.camera) if a.camera.isdigit() else a.camera, a.seconds, a.name)
    else:
        evaluate(a.video, a.box)
