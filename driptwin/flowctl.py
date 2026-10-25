"""Hold a flow on the real rig from the terminal: give mL/h, the servo follows.

    python -m driptwin.flowctl 95                    # hold 95 mL/h until Ctrl+C
    python -m driptwin.flowctl 95 --minutes 10
    python -m driptwin.flowctl 95 --camera 1         # count drops with the webcam itself

While it runs, type a new rate (e.g. 115) and Enter to change the target.

Drops come from a running `driptwin.server --monitor` (its frames carry the drop
counter, so it can keep the camera) or, with --camera, straight from the webcam.
firmware/main.py is loaded into the Pico's RAM, not saved, so the Pico must be at
a blank REPL; on exit the valve is opened and the Pico reboots to that REPL.
The valve's measured curve is valve_cal.json (python -m driptwin.valvecal).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

from .config import CONFIG
from .physics import load_valve
from .twin import DripTwin, Order
from .runtime import load_firmware
from .valvecal import CAL_FILE

class Pico:
    """firmware/main.py over USB serial: set/heartbeat out, status in."""

    def __init__(self, port: str):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=0.1)
        self.status: dict = {}
        self._load_firmware()
        threading.Thread(target=self._reader, daemon=True).start()

    def _load_firmware(self) -> None:
        try:
            load_firmware(self.ser)
        except RuntimeError as e:
            raise SystemExit(str(e))

    def _reader(self) -> None:
        while True:
            try:
                msg = json.loads(self.ser.readline())
            except ValueError:
                continue
            if msg.get("ev") == "status":
                self.status = msg
            elif msg.get("ev") == "alarm" and msg.get("kind") != "boot":
                print(f"  pico alarm: {msg}", flush=True)

    def send(self, u: float) -> None:
        for obj in ({"cmd": "set", "u": round(u, 4)}, {"cmd": "hb"}):
            self.ser.write((json.dumps(obj) + "\n").encode())

    def release(self) -> None:
        """Open the valve, then stop the firmware (its watchdog reboots the Pico)."""
        for _ in range(int(3 / 0.5)):
            self.send(0.0)
            time.sleep(0.5)
        self.ser.write(b"\x03")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rate", type=float, help="target flow, mL/h")
    ap.add_argument("--drop-factor", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--frames", default="http://127.0.0.1:8000/frame.jpg",
                    help="a running monitor server's frame URL (drop counter read off the image)")
    ap.add_argument("--min-interval", type=float, default=0.8,
                    help="with the server's frames: ignore counts closer than this (s), i.e. splash "
                         "ripples; caps the countable rate at 60/this drops/min")
    ap.add_argument("--camera", help="count drops with this webcam instead of a server")
    ap.add_argument("--cal", default=str(CAL_FILE))
    a = ap.parse_args()

    cfg = replace(CONFIG, physics=load_valve(a.cal))
    v = cfg.physics.valve
    twin = DripTwin(cfg, head_cm=v.head_cm)
    lock = threading.Lock()
    t0 = time.monotonic()
    drop_times: list[float] = []

    def on_drop(t: float) -> None:
        with lock:
            twin.on_drop(t - t0)
            drop_times.append(t - t0)

    if a.camera is not None:
        from .vision import CameraCounter
        CameraCounter(on_drop, int(a.camera) if a.camera.isdigit() else a.camera).start()
    else:
        from .framecount import FrameDrops
        FrameDrops(on_drop, a.frames, min_interval_s=a.min_interval).start()   # skip splash ripples

    pico = Pico(a.port)
    with lock:
        twin.start(Order(a.rate, a.drop_factor, 1000, 1000), 0.0)
    print(f"target {a.rate:g} mL/h = {twin.order.target_gtt_min:.1f} drops/min. "
          f"Type a new mL/h + Enter to change it, Ctrl+C to stop.", flush=True)

    pending: list[float] = []

    def keyboard() -> None:
        while True:
            try:
                pending.append(float(input()))
            except ValueError:
                print("  (type a number in mL/h)", flush=True)
            except EOFError:
                return
    threading.Thread(target=keyboard, daemon=True).start()

    seen, next_print, next_min = 0, 0.0, 60.0
    try:
        while not a.minutes or time.monotonic() - t0 < a.minutes * 60:
            t = time.monotonic() - t0
            with lock:
                while pending:
                    new = pending.pop(0)
                    twin.order = replace(twin.order, rate_ml_h=new)
                    twin._settled, twin._lo, twin._hi = False, None, None
                    print(f"--- new target {new:g} mL/h = {twin.order.target_gtt_min:.1f} drops/min",
                          flush=True)
                u = twin.tick(t)
                for e in twin.events[seen:]:
                    print(f"  [{e.t:6.1f}] {e.kind}: {e.message}", flush=True)
                seen = len(twin.events)
                r, tgt, settled = twin.rate, twin.order.target_gtt_min, twin._settled
            pico.send(u)
            if t >= next_min:
                n = sum(1 for d in drop_times if next_min - 60 <= d < next_min)
                print(f"=== minute {next_min / 60:.0f}: {n} drops/min "
                      f"({n / tgt * 100 - 100:+.0f}% of target)", flush=True)
                next_min += 60
            if t >= next_print:
                next_print = t + 5
                deg = v.open_deg + (v.closed_deg - v.open_deg) * u
                print(f"{t:6.0f}s  {r.gtt_min:5.1f} / {tgt:.1f} drops/min  valve {deg:5.1f} deg"
                      f"{'  IN BAND' if settled else ''}  {twin.mode.value}", flush=True)
            time.sleep(cfg.control_period_s)
    except KeyboardInterrupt:
        pass
    print("stopping: opening the valve", flush=True)
    pico.release()


if __name__ == "__main__":
    main()
