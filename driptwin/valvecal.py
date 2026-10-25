"""Measure the pinch valve: servo angle -> drop rate -> flow, on the real rig.

    python -m driptwin.valvecal --port COM5 --angles 0:90:10 --hold 30
    python -m driptwin.valvecal --port COM5 --ws ws://127.0.0.1:8000/ws ...

The servo is driven straight from the Pico REPL (no firmware needed), so this
works on a blank Pico. Drops come from the webcam, or with --ws from an already
running `driptwin.server --monitor`, which then keeps the camera.

Writes valve_cal.json: one row per angle with drops/min and mL/h (measured
drop-size curve for the 20-drop set).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

from .physics import gtt_min_to_ml_h

CAL_FILE = Path(__file__).resolve().parent.parent / "valve_cal.json"
SERVO_PIN = 15


def angle_us(angle: float) -> int:
    return round(500 + 2000 * angle / 180)     # SG90: 0 deg = 500 us, 180 deg = 2500 us


class ReplServo:
    """Servo on GP15 through the MicroPython REPL; stops whatever main.py runs."""

    def __init__(self, port: str):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=0.2)
        self.ser.write(b"\x03\x03\r\n")
        time.sleep(0.3)
        self._line(f"from machine import Pin, PWM; s = PWM(Pin({SERVO_PIN})); s.freq(50)")
        self.angle: float | None = None

    def _line(self, code: str) -> None:
        self.ser.write(code.encode() + b"\r\n")
        time.sleep(0.05)
        self.ser.reset_input_buffer()

    def move(self, angle: float, deg_per_s: float = 30.0) -> None:
        start = angle if self.angle is None else self.angle
        steps = max(1, int(abs(angle - start) / deg_per_s * 20))     # _line takes ~50 ms
        for i in range(1, steps + 1):
            a = start + (angle - start) * i / steps
            self._line(f"s.duty_ns({angle_us(a) * 1000})")
        self.angle = angle


class Drops:
    """Drop timestamps from the local webcam or from a running server's /ws."""

    def __init__(self, camera: int | str | None, ws: str | None, frames: str | None = None,
                 min_interval_s: float = 0.0):
        self.times: list[float] = []
        self.lock = threading.Lock()
        if frames:
            from .framecount import FrameDrops
            FrameDrops(self._add, frames, min_interval_s=min_interval_s).start()
        elif ws:
            threading.Thread(target=lambda: asyncio.run(self._ws(ws)), daemon=True).start()
        else:
            from .vision import CameraCounter
            CameraCounter(self._add, camera).start()

    def _add(self, t: float) -> None:
        with self.lock:
            self.times.append(t)

    async def _ws(self, url: str) -> None:
        import websockets
        last = None
        async with websockets.connect(url, max_size=None) as ws:
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") != "state":
                    continue
                n = m["state"]["drops_counted"]
                if last is not None and n > last:
                    now = time.monotonic()
                    for _ in range(n - last):      # 0.5 s state ticks: good enough over a hold
                        self._add(now)
                last = n

    def between(self, t0: float, t1: float) -> list[float]:
        with self.lock:
            return [t for t in self.times if t0 <= t < t1]


def measure(drops: Drops, hold: float, max_drops: int = 40) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold and len(drops.between(t0, time.monotonic())) < max_drops:
        time.sleep(0.2)
    t1 = time.monotonic()
    got = drops.between(t0, t1)
    # count over the whole hold: near shut, drops come in clusters as the tube
    # sticks and frees, so first-to-last spacing would overstate the rate
    gtt_min = 60.0 * len(got) / (t1 - t0)
    return {"gtt_min": round(gtt_min, 2), "drops": len(got), "secs": round(t1 - t0, 1),
            "ml_h": round(gtt_min_to_ml_h(gtt_min, 20), 1)}


def parse_angles(spec: str) -> list[float]:
    if ":" in spec:
        a, b, step = (float(x) for x in spec.split(":"))
        n = int(round(abs(b - a) / step))
        return [a + (b - a) * i / n for i in range(n + 1)]
    return [float(x) for x in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--camera", default="1")
    ap.add_argument("--ws", help="read drops from a running server, e.g. ws://127.0.0.1:8000/ws")
    ap.add_argument("--frames", help="read the drop counter off a running server's frames, "
                                     "e.g. http://127.0.0.1:8000/frame.jpg")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="with --frames: ignore counts closer than this (s) to the previous one")
    ap.add_argument("--angles", default="0:90:10", help="start:stop:step or a,b,c")
    ap.add_argument("--hold", type=float, default=30.0, help="max seconds measured per angle")
    ap.add_argument("--settle", type=float, default=4.0, help="seconds after a move before counting")
    ap.add_argument("--out", default=str(CAL_FILE))
    ap.add_argument("--head", type=float, help="bag fluid level above the drip end, cm (recorded)")
    a = ap.parse_args()

    cam = int(a.camera) if a.camera.isdigit() else a.camera
    drops = Drops(cam, a.ws, a.frames, a.min_interval)
    servo = ReplServo(a.port)
    if not (a.ws or a.frames):
        time.sleep(12)     # camera warm-up + splash auto-locate
    rows = []
    for ang in parse_angles(a.angles):
        servo.move(ang)
        time.sleep(a.settle)
        r = {"angle": round(ang, 1), **measure(drops, a.hold)}
        rows.append(r)
        print(f"{r['angle']:5.1f} deg  {r['gtt_min']:6.1f} drops/min  {r['ml_h']:6.1f} mL/h  "
              f"({r['drops']} drops in {r['secs']} s)", flush=True)
    Path(a.out).write_text(json.dumps({"head_cm": a.head, "drop_factor": 20,
                                       "measured": time.strftime("%Y-%m-%d %H:%M"),
                                       "rows": rows}, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
