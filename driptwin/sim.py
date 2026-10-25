"""A physical simulator of a gravity IV set, so the whole stack (controller,
agent, dashboard) runs with no hardware — and as a demo backup.

It also renders synthetic drip-chamber frames so the OpenCV detector can be
tested end to end without a webcam.
"""

from __future__ import annotations

import random

import numpy as np

from .config import CONFIG, Config
from .physics import drop_volume_ml, flow_ml_min, ml_h_to_gtt_min


class SimPlant:
    def __init__(self, drop_factor: int = 20, head_cm: float = 80.0,
                 bag_volume_ml: float = 1000.0, cfg: Config = CONFIG, seed: int = 1):
        self.cfg = cfg
        self.drop_factor = drop_factor
        self.mount_head_cm = head_cm      # bag hook height above the site
        self.bag_ml = bag_volume_ml
        self.bag_volume0 = bag_volume_ml
        self.extra_r = 0.0                # kink / occlusion
        self.u = 1.0                      # actual valve position (after slew)
        self.u_cmd = 1.0
        self._accum = 0.0
        self._rng = random.Random(seed)
        self.creep_per_min = 0.0          # clamp relaxation, valve units / minute
        self.stick_u = 0.0                # squashed tube: reopens only after this much play
        self.u_tube = 1.0                 # how closed the tube itself is

    # ---- disturbances (wired to dashboard buttons) ----
    def lower_bag(self, cm: float) -> None:
        self.mount_head_cm -= cm

    def kink(self, on: bool = True) -> None:
        self.extra_r = 1e9 if on else 0.0

    @property
    def head_cm(self) -> float:
        # Fluid surface drops as the bag drains.
        drained = (self.bag_volume0 - self.bag_ml) / 1000.0
        if self.bag_ml <= 0:
            return 0.0
        return self.mount_head_cm - drained * self.cfg.physics.bag_height_cm_per_l

    def command(self, u: float) -> None:
        self.u_cmd = u

    def step(self, t: float, dt: float) -> list[float]:
        """Advance physics by dt; returns drop timestamps that occurred."""
        max_du = self.cfg.safety.max_slew_per_s * dt
        self.u += max(-max_du, min(max_du, self.u_cmd - self.u))
        self.u = min(1.0, max(0.0, self.u + self.creep_per_min * dt / 60.0))
        if self.u > self.u_tube:                      # pinching pushes the tube shut
            self.u_tube = self.u
        elif self.u < self.u_tube - self.stick_u:     # it springs back only past the play
            self.u_tube = self.u + self.stick_u
        q = flow_ml_min(self.head_cm, self.u_tube, self.extra_r, self.cfg.physics)
        dv = min(self.bag_ml, q * dt / 60.0)
        self.bag_ml -= dv
        self._accum += dv
        drops = []
        # drops are the set's real size at this flow (measured curve where we have one)
        drop_vol = drop_volume_ml(ml_h_to_gtt_min(q * 60.0, self.drop_factor), self.drop_factor)
        while self._accum >= drop_vol:
            self._accum -= drop_vol
            # drop detaches at a slightly random point in the step
            drops.append(t - dt * self._rng.random() * 0.5)
        return sorted(drops)


class SyntheticChamber:
    """Renders a drip chamber: a pendant drop grows at the orifice, detaches,
    falls, and splashes into the pool. Grayscale 320x240."""

    SPLASH = ((18, 7, 235), (26, 9, 190), (32, 8, 140))     # crown half-width, height, brightness per frame

    W, H = 240, 320

    def __init__(self, fps: float = 30.0, noise: float = 6.0):
        self.fps = fps
        self.noise = noise
        self._fall: list[float] = []      # y positions of falling drops
        self._splashes: list[int] = []    # frames since each landing
        self._rng = np.random.default_rng(0)

    def frame(self, growth: float, detached: bool) -> np.ndarray:
        """growth in [0,1] = pendant drop size; detached = a drop fell this frame."""
        import cv2
        img = np.full((self.H, self.W), 35, np.uint8)
        cv2.rectangle(img, (60, 10), (180, 310), 70, 2)            # chamber walls
        cv2.rectangle(img, (110, 10), (130, 40), 120, -1)          # orifice
        cv2.rectangle(img, (62, 250), (178, 308), 90, -1)          # fluid pool
        if detached:
            self._fall.append(50.0)
        r = int(3 + 9 * growth)
        cv2.circle(img, (120, 40 + r), r, 210, -1)                 # pendant drop
        v = 900.0 / self.fps                                       # px per frame
        nxt = []
        for y in self._fall:
            y += v
            if y < 250:
                cv2.ellipse(img, (120, int(y)), (6, 9), 0, 0, 360, 200, -1)
                nxt.append(y)
            else:
                self._splashes.append(0)
        self._fall = nxt
        for k in self._splashes:                                   # crown of spray on the surface
            a, b, lum = self.SPLASH[k]
            cv2.ellipse(img, (120, 250), (a, b), 0, 180, 360, lum, -1 if k < 2 else 2)
        self._splashes = [k + 1 for k in self._splashes if k + 1 < len(self.SPLASH)]
        img =img.astype(np.int16) + self._rng.normal(0, self.noise, img.shape).astype(np.int16)
        return np.clip(img, 0, 255).astype(np.uint8)
