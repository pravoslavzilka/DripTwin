"""Wires sensors, the twin, the actuator, storage and the agent together.

Two device backends share one interface:
  SimDevice   physics plant + synthetic chamber video run through the *real*
              OpenCV detector, so the sim exercises the actual CV pipeline.
  PicoDevice  USB serial to firmware/main.py; webcam via vision.CameraCounter.

CameraDevice (monitor mode) is the webcam alone: real drop counting with no
valve, for testing vision before the actuator hardware exists.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import cv2

from dataclasses import replace

from .config import CONFIG
from .physics import load_valve
from .sim import SimPlant, SyntheticChamber
from .storage import Store
from .twin import DripTwin
from .valvecal import CAL_FILE as VALVE_CAL
from .vision import DropDetector, ROI, ROIConfig, CameraCounter


class SimDevice:
    FPS = 30.0

    def __init__(self, on_drop):
        self.on_drop = on_drop
        self.plant = SimPlant()
        self.chamber = SyntheticChamber(fps=self.FPS)
        self.detector = DropDetector(ROIConfig(ROI(85, 225, 70, 40)))
        self.jpeg: bytes | None = None
        self.status = {"u": 1.0, "host": True}
        self._frame_n = 0

    def command(self, u: float) -> None:
        self.plant.command(u)

    def heartbeat(self) -> None:
        pass

    def step(self, t: float, dt: float) -> None:
        drops = self.plant.step(t, dt)
        growth = min(1.0, self.plant._accum * self.plant.drop_factor)
        frame = self.chamber.frame(growth, bool(drops))
        if self.detector.process(frame, t):
            self.on_drop(t)
        self._frame_n += 1
        if self._frame_n % 3 == 0:
            img = cv2.resize(self.detector.annotate(frame), None, fx=1.5, fy=1.5)
            self.jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        self.status = {"u": round(self.plant.u, 3), "host": True,
                       "true_head_cm": round(self.plant.head_cm, 1)}

    # disturbances for the demo panel
    def disturb(self, action: str) -> str:
        p = self.plant
        match action:
            case "lower_bag": p.lower_bag(25)
            case "raise_bag": p.lower_bag(-25)
            case "kink": p.kink(True)
            case "unkink": p.kink(False)
            case "creep": p.creep_per_min = -0.01 if p.creep_per_min == 0 else 0.0
            case "drain": p.bag_ml = min(p.bag_ml, 40)
            case _: return "unknown"
        return action


FIRMWARE = Path(__file__).resolve().parent.parent / "firmware" / "main.py"


def load_firmware(ser) -> None:
    """Paste firmware/main.py into the Pico's RAM through the raw REPL and run it.
    Nothing is saved, so the Pico must be at a blank REPL (no main.py); a reboot
    (its watchdog, after Ctrl-C) brings it back to that REPL."""
    ser.write(b"\r\x03\x03")
    time.sleep(0.2)
    ser.write(b"\x01")
    time.sleep(0.3)
    ser.reset_input_buffer()
    code = FIRMWARE.read_bytes()
    for i in range(0, len(code), 256):
        ser.write(code[i:i + 256])
        time.sleep(0.01)
    ser.write(b"\x04")
    time.sleep(0.5)
    if b"OK" not in ser.read(200):
        raise RuntimeError("Pico did not take the firmware: is it at a blank REPL (no main.py)?")


class PicoDevice:
    def __init__(self, on_drop, port: str, camera: int | str | None = 0):
        import serial
        self.on_drop = on_drop
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.1)
        except serial.SerialException as e:
            raise SystemExit(f"Can't open the Pico on {port}: another program is using it "
                             f"(flowctl, valvecal, servo.py, mpremote, Thonny?). Close it and "
                             f"start again.\n  ({e})") from None
        load_firmware(self.ser)
        self.status: dict = {}
        self.alarms: list[dict] = []
        self.camera = CameraCounter(on_drop, camera) if camera is not None else None
        if self.camera:
            self.camera.start()
        threading.Thread(target=self._reader, daemon=True).start()

    @property
    def jpeg(self):
        return self.camera.jpeg if self.camera else None

    def _send(self, obj: dict) -> None:
        self.ser.write((json.dumps(obj) + "\n").encode())

    def command(self, u: float) -> None:
        self._send({"cmd": "set", "u": round(u, 4)})

    def heartbeat(self) -> None:
        self._send({"cmd": "hb"})

    def step(self, t: float, dt: float) -> None:
        pass

    def disturb(self, action: str) -> str:
        return "real hardware: do it by hand"

    def close(self) -> None:
        """Leave the line as found: open the valve, then stop the firmware."""
        for _ in range(6):
            self.command(0.0)
            self.heartbeat()
            time.sleep(0.5)
        self.ser.write(b"\x03")
        self.ser.close()

    def _reader(self) -> None:
        while True:
            line = self.ser.readline()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            ev = msg.get("ev")
            if ev == "status":
                self.status = msg
            elif ev == "alarm":
                self.alarms.append(msg)


class CameraDevice:
    def __init__(self, on_drop, camera: int | str = 0):
        self.camera = CameraCounter(on_drop, camera)
        self.camera.start()
        self.status: dict = {}

    @property
    def jpeg(self):
        return self.camera.jpeg

    def command(self, u: float) -> None:
        pass

    def heartbeat(self) -> None:
        pass

    def step(self, t: float, dt: float) -> None:
        c = self.camera
        self.status = {"camera_ok": c.alive, "camera_error": c.error, "fps": round(c.fps, 1)}

    def disturb(self, action: str) -> str:
        return "monitor mode: no simulator"


class Runtime:
    def __init__(self, sim: bool = True, port: str | None = None, camera: int | str | None = 0,
                 head_cm: float | None = None, monitor: bool = False):
        self.lock = threading.Lock()
        cfg = CONFIG
        if not sim and not monitor and VALVE_CAL.exists():
            # the real valve: use its measured curve instead of the generic model
            cfg = replace(CONFIG, physics=load_valve(VALVE_CAL))
            v = cfg.physics.valve
            print(f"[runtime] valve calibration {VALVE_CAL.name}: open {v.open_deg:g} deg, "
                  f"shut {v.closed_deg:g} deg, {v.q_ml_min[0] * 60:.0f} mL/h wide open")
        self.twin = DripTwin(cfg, head_cm=head_cm, monitor_only=monitor)
        self.t0 = time.monotonic()
        if sim:
            self.device = SimDevice(self._on_drop)
        elif monitor:
            self.device = CameraDevice(self._on_drop, camera)
        else:
            self.device = PicoDevice(self._on_drop, port, camera)
        self.sim = sim
        self.monitor = monitor
        self.store = Store(physics=cfg.physics)
        self.subscribers: set[asyncio.Queue] = set()
        self._events_sent = 0
        self._fw_alarms_seen = 0

    def now(self) -> float:
        return time.monotonic() - self.t0

    def _on_drop(self, t: float) -> None:
        if not self.sim:
            t -= self.t0      # the webcam thread stamps in monotonic time
        with self.lock:
            self.twin.on_drop(t)
            snap = self.twin.snapshot()
        self.store.drop(snap, t)

    def log(self, kind: str, data: dict) -> None:
        self.store.event(kind, "info", json.dumps(data, default=str), data)

    async def run(self) -> None:
        dt = 1.0 / SimDevice.FPS
        next_tick = next_hb = next_tel = 0.0
        while True:
            t = self.now()
            self.device.step(t, dt)
            if t >= next_tick:
                next_tick = t + CONFIG.control_period_s
                with self.lock:
                    u = self.twin.tick(t)
                    snap = self.twin.snapshot()
                    new = self.twin.events[self._events_sent:]
                    self._events_sent = len(self.twin.events)
                self.device.command(u)
                snap["device"] = self.device.status
                snap["sim"] = self.sim
                for e in new:
                    self.store.event(e.kind, e.severity, e.message, e.data)
                self._fw_alarms(snap)
                self._broadcast({"type": "state", "state": snap,
                                 "events": [e.__dict__ for e in new]})
                if (snap["order"] or self.monitor) and t >= next_tel:
                    next_tel = t + 1.0
                    self.store.telemetry(snap)
            if t >= next_hb:
                next_hb = t + 1.0
                self.device.heartbeat()
            await asyncio.sleep(max(0.0, dt - (self.now() - t)))

    def _fw_alarms(self, snap: dict) -> None:
        alarms = getattr(self.device, "alarms", [])
        for a in alarms[self._fw_alarms_seen:]:
            if a.get("kind") == "boot":
                continue
            msg = {"host_lost": "Controller lost contact with the laptop and is holding "
                                "the clamp where it was."}.get(a["kind"], str(a))
            with self.lock:
                self.twin._emit(self.now(), "fw_" + a["kind"], "alarm", msg, a)
        self._fw_alarms_seen = len(alarms)

    def _broadcast(self, msg: dict) -> None:
        for q in list(self.subscribers):
            if q.qsize() < 20:
                q.put_nowait(msg)
