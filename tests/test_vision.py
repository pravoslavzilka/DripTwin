"""Run the real OpenCV detector on rendered drip-chamber frames with known drops."""
from pathlib import Path

import numpy as np
import pytest

from driptwin.sim import SyntheticChamber
from driptwin.vision import DropDetector, ROI, ROIConfig

ROIS = ROIConfig(ROI(85, 225, 70, 40))


WARMUP_S = 3.5   # detector learns the scene for 90 frames before counting


def render_and_count(period_s, seconds=60, fps=30, noise=6.0):
    ch = SyntheticChamber(fps=fps, noise=noise)
    det = DropDetector(ROIS)
    frames_per_drop = round(period_s * fps)
    true = counted = 0
    for n in range(int(seconds * fps)):
        phase = (n % frames_per_drop) / frames_per_drop
        detached = n > 0 and n % frames_per_drop == 0
        fired = det.process(ch.frame(phase, detached), n / fps)
        if n / fps >= WARMUP_S:
            true += detached
            counted += fired
    return true, counted


@pytest.mark.parametrize("period", [0.4, 0.75, 1.5, 3.0])     # 150, 80, 40, 20 drops/min
def test_counts_match(period):
    true, counted = render_and_count(period)
    assert abs(counted - true) <= 1


def test_noisy_scene():
    true, counted = render_and_count(1.5, noise=14.0)
    assert abs(counted - true) <= 2


def test_no_false_drops_when_static():
    ch, det = SyntheticChamber(), DropDetector(ROIS)
    assert sum(det.process(ch.frame(0.5, False), n / 30) for n in range(600)) == 0


def test_finds_the_splash_box_by_itself():
    ch, det = SyntheticChamber(), DropDetector(None, locate_frames=150)
    n_drop = round(0.75 * 30)
    for n in range(150 + 1):
        det.process(ch.frame((n % n_drop) / n_drop, n > 0 and n % n_drop == 0), n / 30)
    r = det.rois.splash
    assert r.y <= 250 <= r.y + r.h and r.x <= 120 <= r.x + r.w      # where drops land
    true, counted = 0, 0
    for n in range(151, 151 + 60 * 30):
        detached = n % n_drop == 0
        fired = det.process(ch.frame((n % n_drop) / n_drop, detached), n / 30)
        if n / 30 >= 151 / 30 + WARMUP_S:
            true += detached
            counted += fired
    assert abs(counted - true) <= 1


def test_whole_picture_change_is_not_a_drop():
    ch, det = SyntheticChamber(), DropDetector(ROIS)
    frames = [ch.frame(0.5, False) for _ in range(200)]
    for n, f in enumerate(frames):
        det.process(f, n / 30)
    bright = np.clip(frames[-1].astype(int) + 60, 0, 255).astype(np.uint8)   # room light on
    assert not det.process(bright, 200 / 30)


def test_splash_that_flashes_the_whole_chamber_still_counts():
    """On the rig each splash also lights up ~20 % of the picture outside the box
    for a frame or two; that once vetoed nearly every drop as a 'disturbance'.
    Low noise as on the rig, so the flash stands far above the quiet outside."""
    ch, det = SyntheticChamber(noise=2.0), DropDetector(ROIS)
    n_drop, true, counted = round(1.5 * 30), 0, 0
    for n in range(60 * 30):
        detached = n > 0 and n % n_drop == 0
        f = ch.frame((n % n_drop) / n_drop, detached)
        in_box = any(225 <= y < 250 for y in ch._fall)
        if in_box or 1 in ch._splashes:                # the drop reached the box / splashed this frame
            # backlight flash outside the box, ~0.8x the box's own change as measured on the rig
            f = f.copy()
            f[:, :80] = np.clip(f[:, :80].astype(int) + (60 if 1 in ch._splashes else 25), 0, 255)
        fired = det.process(f, n / 30)
        if n / 30 >= WARMUP_S:
            true += detached
            counted += fired
    assert abs(counted - true) <= 1


def test_hand_in_view_is_not_counted():
    """A hand moving through the picture for a second: change outside for many frames."""
    ch, det = SyntheticChamber(), DropDetector(ROIS)
    for n in range(200):
        det.process(ch.frame(0.5, False), n / 30)
    rng, fired = np.random.default_rng(3), 0
    skin = rng.integers(120, 240, (300, 300)).astype(np.uint8)
    for k in range(30):
        f = ch.frame(0.5, False).copy()
        x = 6 * k                                      # sweeps across, over the box too
        f[0:290, x:x + 90] = skin[0:290, k:k + 90][:, :f[0:290, x:x + 90].shape[1]]
        fired += det.process(f, (200 + k) / 30)
    assert fired <= 1                                  # at most its first frame


def test_old_orifice_box_file_still_loads(tmp_path):
    p = tmp_path / "rois.json"
    p.write_text('{"orifice": {"x": 1, "y": 2, "w": 30, "h": 40}}')
    assert ROIConfig.load(p).splash == ROI(1, 2, 30, 40)


REAL_CLIP = Path(__file__).resolve().parent.parent / "recordings" / "brio_splash.mp4"


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="real recording not present")
def test_real_brio_recording_counts_every_splash():
    """40 s from the Brio 101 at ~2 drops/s; reference drops in brio_splash.json."""
    import cv2
    import json
    ref = json.loads(REAL_CLIP.with_suffix(".json").read_text())["drop_frames"]
    cap, det, fired, n = cv2.VideoCapture(str(REAL_CLIP)), DropDetector(None), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if det.process(frame, n / 30):
            fired.append(n)
        n += 1
    start = fired[0] - 3 if fired else n            # counting starts after locate + warm-up
    expected = [r for r in ref if r >= start]
    found = sum(any(abs(f - r) <= 6 for f in fired) for r in expected)
    extra = sum(not any(abs(f - r) <= 6 for r in ref) for f in fired)
    assert found == len(expected) and extra == 0 and len(expected) > 20


def test_fast_drip_keeps_counting():
    """40 s of the box signal from the rig at ~150 drops/min. A falling drop is in
    the box almost all the time, so between splashes the signal sits around 10 and
    splashes reach 27-40; a threshold of 3.5x that baseline counted only ~40 %."""
    from driptwin.vision import SpikeDetector
    data = np.loadtxt(Path(__file__).parent / "data" / "fast_drip_box_signal.csv", delimiter=",")
    t, x = data[:, 0], data[:, 1]
    peaks = sum(1 for i in range(1, len(x) - 1) if x[i] > 20 and x[i] >= x[i - 1] and x[i] >= x[i + 1])
    det = SpikeDetector(warmup=45)                    # 15 fps: the same 3 s of learning
    fired = sum(det.update(v, tt) for v, tt in zip(x, t))
    assert fired >= 0.8 * peaks
