"""Hydraulic model of a gravity IV set and the online 'digital twin' estimator.

The trick that makes DripTwin explainable: given the valve position we
commanded and the drop rate we measure, the model back-solves the *implied
head* H. Each failure mode leaves a different fingerprint on H:

  bag lowered/raised   -> H steps by the height change, flow responds to valve
  bag draining         -> H creeps down a few cm per litre
  kink / occlusion     -> flow collapses; opening the valve does not bring it back
  empty bag            -> H falls toward P_ven as infused volume -> bag volume
"""

from dataclasses import dataclass
import math

from .config import PhysicsParams, CONFIG


@dataclass(frozen=True)
class DropCurve:
    """Measured drop size of one tubing set: drops grow as the drip speeds up, so a
    fixed 1/drop_factor mL per drop under-reads fast flows.

        V(f) = v0 + dv * f^n / (f^n + f50^n)   mL per drop, f in drops/s
        Q(f) = f * V(f)                         mL/s
    """
    v0: float
    dv: float
    f50: float
    n: float = 4.0

    def volume_ml(self, f: float) -> float:
        f = max(0.0, f)
        return self.v0 + self.dv * f ** self.n / (f ** self.n + self.f50 ** self.n)

    def flow_ml_s(self, f: float) -> float:
        return max(0.0, f) * self.volume_ml(f)

    def freq_for_flow(self, q_ml_s: float) -> float:
        """Inverse of Q(f). Q only rises with f (V never shrinks), so bisect."""
        if q_ml_s <= 0:
            return 0.0
        lo, hi = 0.0, q_ml_s / self.v0           # V >= v0, so f <= q / v0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.flow_ml_s(mid) < q_ml_s else (lo, mid)
        return (lo + hi) / 2


# Fitted to our bench measurements of the 20 gtt/mL set (drop rate vs collected volume).
# Other sets use the nominal 1/drop_factor mL per drop.
DROP_CURVES: dict[int, DropCurve] = {
    20: DropCurve(v0=0.06341, dv=0.02416, f50=2.688, n=4.0),
}


def drop_volume_ml(gtt_min: float, drop_factor: int) -> float:
    """Volume of one drop at this drip rate."""
    curve = DROP_CURVES.get(drop_factor)
    return curve.volume_ml(gtt_min / 60.0) if curve else 1.0 / drop_factor


def ml_h_to_gtt_min(rate_ml_h: float, drop_factor: int) -> float:
    curve = DROP_CURVES.get(drop_factor)
    if curve:
        return 60.0 * curve.freq_for_flow(rate_ml_h / 3600.0)
    return rate_ml_h * drop_factor / 60.0


def gtt_min_to_ml_h(gtt_min: float, drop_factor: int) -> float:
    return gtt_min * drop_volume_ml(gtt_min, drop_factor) * 60.0


def _conductance(u: float, v) -> float:
    """Measured valve: flow per cmH2O of driving pressure, linear between table rows."""
    dp = v.head_cm - CONFIG.physics.p_venous_cmh2o
    us, qs = v.u, v.q_ml_min
    u = min(us[-1], max(us[0], u))
    for i in range(1, len(us)):
        if u <= us[i]:
            w = (u - us[i - 1]) / (us[i] - us[i - 1])
            return (qs[i - 1] + w * (qs[i] - qs[i - 1])) / dp
    return qs[-1] / dp


def clamp_resistance(u: float, p: PhysicsParams = CONFIG.physics) -> float:
    if p.valve:
        g = _conductance(u, p.valve)
        return math.inf if g <= 0 else max(0.0, 1.0 / g - p.r0)
    if u >= p.u_occlude:
        return math.inf
    gap = 1.0 - max(0.0, u)
    return p.kc * (1.0 / gap**3 - 1.0)


def flow_ml_min(head_cm: float, u: float, extra_r: float = 0.0,
                p: PhysicsParams = CONFIG.physics) -> float:
    """Forward model. extra_r models a kink or partial occlusion."""
    dp = head_cm * p.rho_g - p.p_venous_cmh2o
    if dp <= 0:
        return 0.0
    r = p.r0 + clamp_resistance(u, p) + extra_r
    return 0.0 if math.isinf(r) else dp / r


def implied_head(q_ml_min: float, u: float, p: PhysicsParams = CONFIG.physics) -> float:
    """Inverse model: what head would produce this flow at this valve position?"""
    rc = clamp_resistance(u, p)
    if math.isinf(rc):
        return math.nan
    return (q_ml_min * (p.r0 + rc) + p.p_venous_cmh2o) / p.rho_g


def valve_for_flow(head_cm: float, q_target_ml_min: float,
                   p: PhysicsParams = CONFIG.physics) -> float:
    """Model inverse: valve position that yields q_target at the given head.
    Returns 0.0 (wide open) if the target is unreachable."""
    dp = head_cm * p.rho_g - p.p_venous_cmh2o
    if q_target_ml_min <= 0:
        return 1.0
    if dp <= 0:
        return 0.0
    if p.valve:
        # walk the (decreasing) measured curve to the target conductance
        g, v = q_target_ml_min / dp, p.valve
        us = v.u
        if g >= _conductance(us[0], v):
            return 0.0
        for i in range(1, len(us)):
            g0, g1 = _conductance(us[i - 1], v), _conductance(us[i], v)
            if g1 <= g:
                return us[i - 1] + (us[i] - us[i - 1]) * (g0 - g) / (g0 - g1)
        return us[-1]
    rc = dp / q_target_ml_min - p.r0
    if rc <= 0:
        return 0.0
    gap = (p.kc / (rc + p.kc)) ** (1.0 / 3.0)
    return min(p.u_occlude - 1e-3, max(0.0, 1.0 - gap))


def max_flow_ml_min(head_cm: float, p: PhysicsParams = CONFIG.physics) -> float:
    return flow_ml_min(head_cm, 0.0, p=p)


@dataclass
class TwinState:
    head_cm: float              # filtered implied head
    head_baseline_cm: float     # head at last 'settled' point
    head_fast_cm: float         # lightly filtered
    infused_ml: float
    bag_volume_ml: float
    head_inst_cm: float = 0.0   # latest unfiltered implied head (step detection)

    @property
    def remaining_ml(self) -> float:
        return max(0.0, self.bag_volume_ml - self.infused_ml)


class HeadEstimator:
    """Tracks implied head with a fast and slow EWMA. Only updates while the
    rate estimate is trustworthy and the valve is not occluding."""

    def __init__(self, initial_head_cm: float, bag_volume_ml: float,
                 p: PhysicsParams = CONFIG.physics):
        self.p = p
        self.state = TwinState(initial_head_cm, initial_head_cm, initial_head_cm,
                               0.0, bag_volume_ml, initial_head_cm)

    def update(self, q_ml_min: float, u: float, dt: float, confident: bool) -> TwinState:
        s = self.state
        s.infused_ml += q_ml_min * dt / 60.0
        if confident and q_ml_min > 0:
            h = implied_head(q_ml_min, u, self.p)
            if not math.isnan(h):
                s.head_inst_cm = h
                s.head_fast_cm += min(1.0, dt / 3.0) * (h - s.head_fast_cm)
                s.head_cm += min(1.0, dt / 12.0) * (h - s.head_cm)
        return s

    def rebaseline(self) -> None:
        self.state.head_baseline_cm = self.state.head_fast_cm
        self.state.head_cm = self.state.head_fast_cm


def _decreasing_fit(ys: list[float]) -> list[float]:
    """Closest non-increasing sequence (pool adjacent violators): readings that
    break the trend are averaged with their neighbours instead of clipped."""
    blocks: list[list[float]] = []          # [mean, count]
    for y in ys:
        blocks.append([y, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            (m2, n2), (m1, n1) = blocks.pop(), blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    return [m for m, n in blocks for _ in range(n)]


def load_valve(path, base: PhysicsParams = CONFIG.physics) -> PhysicsParams:
    """PhysicsParams for the measured valve in valve_cal.json (driptwin.valvecal).

    The useful range is from the angle of most flow to the first angle that stops
    it. Flow is forced non-increasing across it (drop counts are noisy), and r0 is
    set so the model's wide-open flow is the measured one."""
    import json
    from dataclasses import replace
    from pathlib import Path
    from .config import ValveTable
    cal = json.loads(Path(path).read_text())
    rows = sorted(cal["rows"], key=lambda r: r["angle"])
    if cal.get("open_deg") is not None:
        open_deg, closed_deg = cal["open_deg"], cal["closed_deg"]
    else:
        best = max(rows, key=lambda r: r["ml_h"])
        least = min(rows, key=lambda r: r["ml_h"])
        side = sorted((r for r in rows if (r["angle"] - best["angle"]) *
                       (least["angle"] - best["angle"]) > 0),
                      key=lambda r: abs(r["angle"] - best["angle"]))   # walking toward shut
        shut = next((r for r in side if r["ml_h"] < 1.0), least)
        open_deg, closed_deg = best["angle"], shut["angle"]
    lo, hi = sorted((open_deg, closed_deg))
    span = [r for r in rows if lo <= r["angle"] <= hi]
    if open_deg > closed_deg:
        span.reverse()
    us = [(r["angle"] - open_deg) / (closed_deg - open_deg) for r in span]
    qs = _decreasing_fit([r["ml_h"] / 60.0 for r in span])
    head = cal.get("head_cm") or CONFIG.default_head_cm
    backlash = cal.get("backlash_deg", 0.0) / abs(closed_deg - open_deg)
    valve = ValveTable(tuple(us), tuple(qs), head, open_deg, closed_deg, backlash,
                       settle_s=cal.get("settle_s", 1.0), rate_window=cal.get("rate_window", 0),
                       stall_s=cal.get("stall_s", 0.0), head_alert_cm=cal.get("head_alert_cm", 10.0),
                       step_deg=cal.get("step_deg", 1.0))
    return replace(base, valve=valve, r0=(head - base.p_venous_cmh2o) / qs[0])
