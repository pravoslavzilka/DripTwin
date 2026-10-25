"""Control with a measured valve table (driptwin.valvecal) instead of the generic model."""
import json
from dataclasses import replace

import pytest

from scenario import run
from driptwin.config import CONFIG
from driptwin.physics import flow_ml_min, load_valve, ml_h_to_gtt_min, valve_for_flow
from driptwin.sim import SimPlant
from driptwin.twin import DripTwin, Order

# a pinch valve as it measures: flat far from the tube, then steep near shut
ANGLES = [0, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 90]
ML_H = [400, 398, 396, 390, 380, 360, 320, 260, 170, 70, 0, 0]


def cal_file(tmp_path, angles=ANGLES, ml_h=ML_H, **extra):
    f = tmp_path / "valve_cal.json"
    f.write_text(json.dumps({"head_cm": 80, "rows": [{"angle": a, "ml_h": q}
                                                     for a, q in zip(angles, ml_h)], **extra}))
    return f


def test_table_is_read_and_inverted(tmp_path):
    p = load_valve(cal_file(tmp_path))
    v = p.valve
    assert (v.open_deg, v.closed_deg) == (0, 80)
    assert flow_ml_min(80, 0.0, p=p) == pytest.approx(400 / 60)
    assert flow_ml_min(80, 1.0, p=p) == 0.0
    u70 = (70 - 0) / 80
    assert flow_ml_min(80, u70, p=p) == pytest.approx(170 / 60)
    assert valve_for_flow(80, 170 / 60, p) == pytest.approx(u70)
    for q in (1.0, 3.0, 4.5):                      # round trip at another head too
        assert flow_ml_min(60, valve_for_flow(60, q, p), p=p) == pytest.approx(q, rel=1e-6)


def test_noisy_table_is_made_monotonic(tmp_path):
    p = load_valve(cal_file(tmp_path, ml_h=[400, 405, 396, 390, 395, 360, 320, 260, 170, 70, 0, 0]))
    assert list(p.valve.q_ml_min) == sorted(p.valve.q_ml_min, reverse=True)


def test_reversed_valve_direction(tmp_path):
    p = load_valve(cal_file(tmp_path, ml_h=list(reversed(ML_H))))
    assert (p.valve.open_deg, p.valve.closed_deg) == (90, 10)


@pytest.mark.parametrize("shift", [0.0, 0.04])     # 0.04 = real valve 3 deg off the table
def test_reaches_target_band_with_measured_valve(tmp_path, shift):
    p = load_valve(cal_file(tmp_path))
    cfg = replace(CONFIG, physics=p)
    real = replace(p, valve=replace(p.valve, u=tuple(min(1.0, u + shift) if 0 < u < 1 else u
                                                     for u in p.valve.u)))
    plant = SimPlant(cfg=replace(CONFIG, physics=real))
    twin, _, log = run(120, plant=plant, twin=DripTwin(cfg, head_cm=80.0))
    target = ml_h_to_gtt_min(120, 20)
    late = [r[1] for r in log if r[0] > 70]
    assert sum(late) / len(late) == pytest.approx(target, rel=cfg.band_out)
    assert twin.mode.value == "running"


def test_sticky_tube_is_approached_from_open(tmp_path):
    """The rig's tube reopens only after extra travel. Opening moves overshoot by
    backlash_deg and close back, so the valve lands where the table says even
    when the bag is lowered (open) and raised again (close)."""
    p = load_valve(cal_file(tmp_path, backlash_deg=6))
    cfg = replace(CONFIG, physics=p)
    plant = SimPlant(cfg=cfg)
    plant.stick_u = 6 / 80
    twin, _, log = run(400, [(120, lambda pl: pl.lower_bag(20)), (260, lambda pl: pl.lower_bag(-20))],
                       plant=plant, twin=DripTwin(cfg, head_cm=80.0))
    target = ml_h_to_gtt_min(120, 20)
    for a, b in ((60, 120), (200, 260), (340, 400)):
        rows = [r[1] for r in log if a < r[0] < b]
        assert sum(rows) / len(rows) == pytest.approx(target, rel=cfg.band_out)
    assert "probe" not in [e.kind for e in twin.events]


def test_valve_moves_wait_for_fresh_drops(tmp_path):
    """Each correction waits for the settle time plus a full window of new drops,
    even when the model asks for a tiny step (they once crept every tick)."""
    # a cliff like the rig's: on it, a 6.5 % flow error asks for a ~0.05 deg step
    p = load_valve(cal_file(tmp_path, angles=[0, 60, 61, 62, 63], ml_h=[400, 400, 300, 100, 0],
                            settle_s=8.0, rate_window=16, step_deg=0.5))
    twin = DripTwin(replace(CONFIG, physics=p), head_cm=80.0)
    twin.start(Order(200, 20, 1000, 1000), 0.0)
    moves, last_u, t, nd = [], twin.u, 0.0, 0.0
    while t < 400:
        t = round(t + 0.5, 3)
        while nd < t:
            twin.on_drop(nd)
            nd += 60 / (ml_h_to_gtt_min(200, 20) * 1.065)     # steadily 6.5 % fast
        u = twin.tick(t)
        if abs(u - last_u) > 1e-9 and t > 20:
            moves.append(t)
        last_u = u
    gaps = [b - a for a, b in zip(moves, moves[1:])]
    assert len(moves) <= 2 or min(gaps) >= 8.0 + 15 * 60 / (ml_h_to_gtt_min(200, 20) * 1.065)


def test_converges_when_rig_drifts_off_table(tmp_path):
    """Bag 50 % higher than at calibration and the tube pinching 2 deg later:
    the controller still gets into the band within a few minutes."""
    angles = [0, 50, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68]
    ml_h = [150, 150, 150, 104, 104, 104, 93, 93, 93, 21, 21, 0]
    p = load_valve(cal_file(tmp_path, angles=angles, ml_h=ml_h, settle_s=8.0, rate_window=16,
                            step_deg=0.5, stall_s=25.0, backlash_deg=1.5))
    real = replace(p, valve=replace(p.valve, u=tuple(min(1.0, u + 2 / 68) if 0 < u < 1 else u
                                                     for u in p.valve.u)))
    plant = SimPlant(cfg=replace(CONFIG, physics=real), head_cm=120.0)
    twin, _, log = run(600, plant=plant, order=Order(95, 20, 1000, 1000),
                       twin=DripTwin(replace(CONFIG, physics=p), head_cm=80.0))
    target = ml_h_to_gtt_min(95, 20)
    late = [r[1] for r in log if r[0] > 420]
    assert sum(late) / len(late) == pytest.approx(target, rel=0.08)


def test_open_valve_above_target_is_not_free_flow(tmp_path):
    """The bag gives far more than the target wide open: the controller must close
    in on it, not call it free flow (it once did, right after start-up)."""
    p = load_valve(cal_file(tmp_path, settle_s=8.0, rate_window=16, step_deg=0.5, backlash_deg=1.5))
    plant = SimPlant(cfg=replace(CONFIG, physics=p), head_cm=110.0)       # ~140 drops/min wide open
    twin, _, log = run(600, plant=plant, order=Order(330, 20, 1000, 1000),
                       twin=DripTwin(replace(CONFIG, physics=p), head_cm=80.0))
    assert "free_flow" not in [e.kind for e in twin.events]
    late = [r[1] for r in log if r[0] > 420]
    assert sum(late) / len(late) == pytest.approx(ml_h_to_gtt_min(330, 20), rel=0.08)



@pytest.mark.parametrize("stick_deg", [0, 2, 5])
def test_sticky_rig_settles_by_feedback(tmp_path, stick_deg):
    """The rig's curve (flat, then a cliff near 68 deg), the real valve 0.6 deg off
    the table, and a tube that clings by up to 5 deg. Approaching from the open
    side, learning the hysteresis and holding the better end of a too-fine bracket
    get it within the band; a fixed calibration alone did not."""
    angles = [40, 55, 60, 62, 64, 66, 67, 68, 70, 90]
    ml_h = [761, 734, 391, 376, 357, 391, 391, 152, 137, 0]
    p = load_valve(cal_file(tmp_path, angles=angles, ml_h=ml_h, open_deg=40, closed_deg=90,
                            backlash_deg=1.5, settle_s=8.0, rate_window=16, step_deg=0.5, stall_s=25.0))
    real = replace(p, valve=replace(p.valve, u=tuple(min(1.0, u + 0.6 / 50) if 0 < u < 1 else u
                                                     for u in p.valve.u)))
    plant = SimPlant(cfg=replace(CONFIG, physics=real))
    plant.stick_u = stick_deg / 50
    twin, _, log = run(720, plant=plant, order=Order(330, 20, 1000, 1000),
                       twin=DripTwin(replace(CONFIG, physics=p), head_cm=80.0))
    late = [r[1] for r in log if r[0] > 480]
    assert sum(late) / len(late) == pytest.approx(ml_h_to_gtt_min(330, 20), rel=0.08)
    assert not [e for e in twin.events if e.severity == "alarm"]
