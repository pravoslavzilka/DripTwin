import pytest

from scenario import run
from driptwin.physics import (DROP_CURVES, flow_ml_min, gtt_min_to_ml_h, implied_head, valve_for_flow,
                              ml_h_to_gtt_min)
from driptwin.sim import SimPlant
from driptwin.twin import DripTwin, Order, OrderRejected, validate_order


def kinds(twin):
    return [e.kind for e in twin.events]


def settled_rate(log, after):
    rows = [r for r in log if r[0] > after]
    return sum(r[1] for r in rows) / len(rows)


def test_order_conversion():
    # 20 gtt/mL set: measured drops are ~0.063 mL, not the nominal 0.05
    assert ml_h_to_gtt_min(120, 20) == pytest.approx(31.53, abs=0.01)
    assert ml_h_to_gtt_min(100, 60) == pytest.approx(100.0)      # other sets: nominal
    assert ml_h_to_gtt_min(120, 15) == pytest.approx(30.0)


@pytest.mark.parametrize("f", [0.1, 0.5, 1.0, 2.0, 2.688, 5.0])
def test_measured_drop_curve_matches_bench_fit(f):
    c = DROP_CURVES[20]
    assert c.volume_ml(f) == pytest.approx(0.06341 + 0.02416 * f**4 / (f**4 + 2.688**4))
    assert c.flow_ml_s(f) == pytest.approx(0.06341 * f + 0.02416 * f**5 / (f**4 + 2.688**4))
    assert c.freq_for_flow(c.flow_ml_s(f)) == pytest.approx(f, rel=1e-9)
    assert gtt_min_to_ml_h(60 * f, 20) == pytest.approx(3600 * c.flow_ml_s(f))


def test_monitor_mode_reports_flow_from_drop_sizes():
    twin = DripTwin(monitor_only=True)
    t = 0.0
    while t < 30:
        twin.on_drop(t)
        twin.tick(t)
        t += 0.5                                    # 2 drops/s
    s = twin.snapshot()
    assert s["drop_factor"] == 20
    assert s["rate_ml_h"] == pytest.approx(3600 * DROP_CURVES[20].flow_ml_s(2.0), rel=0.02)
    assert s["infused_ml"] == pytest.approx(60 * 0.0691, rel=0.05)   # 60 drops of ~0.069 mL


def test_model_inverse_roundtrip():
    for head in (50, 80, 120):
        for q in (0.5, 2.0, 5.0):
            u = valve_for_flow(head, q)
            assert flow_ml_min(head, u) == pytest.approx(q, rel=1e-3)
            assert implied_head(q, u) == pytest.approx(head, rel=1e-3)


@pytest.mark.parametrize("bad", [Order(1000, 20, 500, 1000), Order(120, 13, 500, 1000),
                                 Order(120, 20, 2000, 1000)])
def test_unsafe_orders_rejected(bad):
    with pytest.raises(OrderRejected):
        validate_order(bad)


def test_reaches_target():
    twin, _, log = run(60)
    assert settled_rate(log, 20) == pytest.approx(ml_h_to_gtt_min(120, 20), rel=0.04)
    assert kinds(twin) == ["order_started"]


def test_bag_lowered_is_diagnosed_and_corrected():
    twin, _, log = run(180, [(60, lambda p: p.lower_bag(25))])
    ev = [e for e in twin.events if e.kind == "head_change"]
    assert len(ev) == 1
    assert ev[0].data["delta_cm"] == pytest.approx(-25, abs=3)
    assert settled_rate(log, 120) == pytest.approx(ml_h_to_gtt_min(120, 20), rel=0.05)


def test_kink_raises_occlusion_and_restores_valve():
    twin, plant, _ = run(120, [(60, lambda p: p.kink(True))])
    assert "probe" in kinds(twin) and "occlusion" in kinds(twin)
    assert "empty_bag" not in kinds(twin)
    probe = next(e for e in twin.events if e.kind == "probe")
    # valve went back to its pre-probe position: no bolus when the kink clears
    assert twin.u > 0.5 and abs(twin.u - plant.u_cmd) < 1e-9
    assert probe.t > 60


def test_brief_pinch_clears_without_alarm():
    twin, _, _ = run(150, [(60, lambda p: p.kink(True)), (70, lambda p: p.kink(False))])
    assert "occlusion" not in kinds(twin)


def test_empty_bag_is_not_called_a_kink():
    plant = SimPlant(bag_volume_ml=80)
    twin, _, _ = run(3600, order=Order(120, 20, 100, 100), plant=plant)
    assert "empty_bag" in kinds(twin)
    assert "occlusion" not in kinds(twin)
    assert twin.u == 1.0


def test_bag_too_low_asks_to_raise_it():
    twin, _, _ = run(120, [(40, lambda p: p.lower_bag(60))], order=Order(400, 20, 1000, 1000))
    assert "low_head" in kinds(twin)


def test_clamp_creep_is_compensated():
    plant = SimPlant()
    plant.creep_per_min = -0.01   # tubing relaxes open -> flow would rise
    _, _, log = run(600, plant=plant)
    assert settled_rate(log, 300) == pytest.approx(ml_h_to_gtt_min(120, 20), rel=0.06)


def test_vtbi_completion_closes_valve():
    twin, _, _ = run(1000, order=Order(300, 20, 50, 1000))
    assert twin.mode.value == "complete" and twin.u == 1.0


def test_monitor_mode_narrates_drops():
    twin = DripTwin(monitor_only=True)
    with pytest.raises(OrderRejected):
        twin.start(Order(120, 20, 1000, 1000), 0.0)
    t, period = 0.0, 1.5                       # 40 drops/min
    while t < 60:
        t = round(t + 0.1, 6)
        if t >= 40:                           # flow speeds up to 60 drops/min
            period = 1.0
        if abs(t / period - round(t / period)) < 1e-6 and t < 90:
            twin.on_drop(t)
        twin.tick(t)
    for _ in range(150):                      # then drops stop for 15 s
        t = round(t + 0.1, 6)
        twin.tick(t)
    assert kinds(twin) == ["drops_detected", "rate_change", "drops_stopped"]
    change = twin.events[1].data
    assert change["from"] == pytest.approx(40, rel=0.05) and change["to"] == pytest.approx(60, rel=0.05)


def test_too_much_flow_is_spoken_with_the_bag_height_excess():
    """The valve isn't holding (tube out of the pinch): flow stays ~40 % over target.
    The twin says so, with how much higher the bag would have to be by the model."""
    plant = SimPlant(head_cm=50.0)
    plant.command = lambda u: None            # valve has no effect
    plant.u = plant.u_cmd = 0.0               # tube fully open
    twin, _, _ = run(240, plant=plant, order=Order(360, 20, 1000, 1000))
    ev = [e for e in twin.events if e.kind == "high_flow"]
    assert len(ev) == 1                        # once, then not again within 2 minutes
    assert ev[0].data["excess_head_cm"] > 5
    assert "higher than it should be" in ev[0].data["say"]["en"]
    assert "más alta" in ev[0].data["say"]["es"]


def test_no_drops_says_whether_the_valve_or_the_line_is_closed():
    twin, _, _ = run(120, [(60, lambda p: p.kink(True))])
    said = [e.data["say"]["en"] for e in twin.events if e.kind == "no_flow"]
    assert said and "although the valve is open" in said[0]       # a kink, not the valve

    twin = DripTwin(head_cm=80.0)
    twin.start(Order(120, 20, 1000, 1000), 0.0)
    twin.u = 1.0                                                  # valve shut
    twin._say_no_flow(30.0, twin.order, 25.0)
    assert twin.events[-1].data["say"]["en"].startswith("Nothing is dripping. The valve is closed")
