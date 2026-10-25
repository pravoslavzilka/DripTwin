"""The DripTwin core: order -> model-inverse controller -> diagnostics.

Runs a fixed-period tick. Each tick:
  1. estimate drop rate (fresh drops only, since the last valve move)
  2. update the digital twin's implied head
  3. classify what is going on (head change, occlusion, empty bag, ...)
  4. choose a valve position (model inverse, slew-limited)

Pure logic, no I/O: the same class drives the simulator, the real Pico,
and the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import math

from .config import CONFIG, Config
from .physics import (DROP_CURVES, HeadEstimator, clamp_resistance, drop_volume_ml,
                      flow_ml_min, gtt_min_to_ml_h, ml_h_to_gtt_min, valve_for_flow,
                      max_flow_ml_min)
from .rate import DropLog, RateEstimator, RateEstimate


class Mode(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PROBING = "probing"            # testing whether opening the valve restores flow
    PAUSED = "paused"
    COMPLETE = "complete"
    ALARM_OCCLUSION = "alarm_occlusion"
    ALARM_EMPTY = "alarm_empty"
    ALARM_FREE_FLOW = "alarm_free_flow"
    ALARM_LOW_HEAD = "alarm_low_head"


ALARMS = {Mode.ALARM_OCCLUSION, Mode.ALARM_EMPTY, Mode.ALARM_FREE_FLOW}


@dataclass
class Order:
    rate_ml_h: float
    drop_factor: int
    vtbi_ml: float          # volume to be infused
    bag_volume_ml: float
    drug: str = "fluid"

    @property
    def target_gtt_min(self) -> float:
        return ml_h_to_gtt_min(self.rate_ml_h, self.drop_factor)

    @property
    def target_ml_min(self) -> float:
        return self.rate_ml_h / 60.0


@dataclass
class Event:
    t: float
    kind: str
    severity: str           # info | warn | alarm
    message: str
    data: dict = field(default_factory=dict)


class OrderRejected(ValueError):
    pass


def validate_order(order: Order, cfg: Config = CONFIG) -> None:
    s = cfg.safety
    if not (s.min_rate_ml_h <= order.rate_ml_h <= s.max_rate_ml_h):
        raise OrderRejected(
            f"rate {order.rate_ml_h} mL/h outside {s.min_rate_ml_h}-{s.max_rate_ml_h} mL/h")
    if order.drop_factor not in s.allowed_drop_factors:
        raise OrderRejected(
            f"drop factor {order.drop_factor} not in {s.allowed_drop_factors} gtt/mL")
    if not (0 < order.vtbi_ml <= min(order.bag_volume_ml, s.max_volume_ml)):
        raise OrderRejected(
            f"VTBI {order.vtbi_ml} mL must be >0 and <= bag volume {order.bag_volume_ml} mL")


class DripTwin:
    def __init__(self, cfg: Config = CONFIG, head_cm: float | None = None,
                 monitor_only: bool = False):
        self.cfg = cfg
        self.monitor_only = monitor_only   # webcam only, no valve: count drops, never infuse
        self.drop_log = DropLog()
        v = cfg.physics.valve
        self.rater = (RateEstimator(v.rate_window, mean=True) if v and v.rate_window
                      else RateEstimator(cfg.rate_window_drops))
        self.head0 = head_cm or cfg.default_head_cm
        self.est = HeadEstimator(self.head0, 1000.0, cfg.physics)
        self.order: Order | None = None
        self.mode = Mode.IDLE
        self.u = 1.0                     # start closed
        self.t = 0.0
        self.t_move = 0.0                # last significant valve move
        self.t_started = 0.0
        self.events: list[Event] = []
        self.rate = RateEstimate(0.0, False, math.inf, 0)
        self._probe: dict | None = None
        self._unstick: tuple[float, float] | None = None   # (target u, when to go there)
        self._settled = False
        self._ever_settled = False
        self._high_since: float | None = None     # flow well above target since (s)
        self._lo = self._hi = None        # valve search bracket (see _search)
        self._push, self._push_dir = 0, False
        self._opened: float | None = None     # flow before the last opening move
        self._lo_err = self._hi_err = 0.0
        v = cfg.physics.valve                 # hysteresis allowance, learned while running
        self._backlash = v.backlash_u if v else 0.0
        self._history: list[dict] = []
        self._flowing, self._ref_rate = False, 0.0     # monitor-mode narration state
        self.monitor_ml = 0.0            # monitor mode: volume dripped, drop by drop

    # ---- inputs -----------------------------------------------------------
    def on_drop(self, t: float) -> None:
        self.drop_log.add(t)
        if self.monitor_only:            # each drop adds its own size at the current rate
            self.monitor_ml += drop_volume_ml(self.rate.gtt_min, self.cfg.monitor_drop_factor)

    def start(self, order: Order, t: float) -> None:
        if self.monitor_only:
            raise OrderRejected("monitor-only mode: no valve is connected, so drops are "
                                "counted but an infusion can't be run")
        validate_order(order, self.cfg)
        self.order = order
        self.est = HeadEstimator(self.head0, order.bag_volume_ml, self.cfg.physics)
        self.mode = Mode.RUNNING
        self.t_started = t
        self._settled = self._ever_settled = False
        self._lo = self._hi = None
        # Feed-forward from the model: start where physics says we should be.
        self._move(valve_for_flow(self.head0, order.target_ml_min, self.cfg.physics), t)
        self._emit(t, "order_started", "info",
                   f"Started {order.rate_ml_h:g} mL/h with a {order.drop_factor} gtt/mL set "
                   f"= {order.target_gtt_min:.1f} drops/min, VTBI {order.vtbi_ml:g} mL."
                   + (f" Using the measured drop size for this set "
                      f"({drop_volume_ml(order.target_gtt_min, order.drop_factor):.3f} mL "
                      f"per drop at this rate)." if order.drop_factor in DROP_CURVES else ""),
                   asdict(order))

    def pause(self, t: float) -> None:
        if self.order:
            self.mode = Mode.PAUSED
            self._move(1.0, t)
            self._emit(t, "paused", "info", "Infusion paused, valve closed.")

    def resume(self, t: float) -> None:
        if self.order and self.mode in (Mode.PAUSED, *ALARMS, Mode.ALARM_LOW_HEAD):
            self.mode = Mode.RUNNING
            self._probe = None
            self._settled = False
            self._move(valve_for_flow(self.est.state.head_cm, self.order.target_ml_min,
                                      self.cfg.physics), t)
            self._emit(t, "resumed", "info", "Infusion resumed.")

    def set_head_hint(self, head_cm: float, t: float) -> None:
        """Nurse tells us the bag height; resets the twin's prior."""
        self.est.state.head_cm = self.est.state.head_fast_cm = head_cm
        self.est.state.head_inst_cm = head_cm
        self.est.state.head_baseline_cm = head_cm
        self._emit(t, "head_hint", "info", f"Bag height set to {head_cm:.0f} cm above the site.")

    # ---- main tick --------------------------------------------------------
    def tick(self, t: float) -> float:
        dt = t - self.t if self.t else self.cfg.control_period_s
        self.t = t
        self.rate = self.rater.estimate(self.drop_log.drops, t, since=self.t_move)
        if self.rate.changepoint is not None and self.rate.changepoint > self.t_move:
            # Flow stepped: forget pre-step intervals so the twin sees the new regime only.
            self.t_move = self.rate.changepoint
            self.rate = self.rater.estimate(self.drop_log.drops, t, since=self.t_move)
        if self.monitor_only:
            self._narrate(t)
            return self.u
        o = self.order
        if o is None or self.mode in (Mode.IDLE, Mode.PAUSED, Mode.COMPLETE):
            return self.u

        q = gtt_min_to_ml_h(self.rate.gtt_min, o.drop_factor) / 60.0   # mL/min
        s = self.est.update(q, self.u, dt, self.rate.confident)
        self._record(t, q)

        if s.infused_ml >= o.vtbi_ml:
            self.mode = Mode.COMPLETE
            self._move(1.0, t)
            self._emit(t, "complete", "info",
                       f"VTBI reached: {s.infused_ml:.0f} mL infused. Valve closed.")
            return self.u

        if self.mode in ALARMS:
            return self.u
        if self._unstick:
            if t < self._unstick[1]:
                return self.u                  # releasing the tube: nothing to judge yet
            self._move(self._unstick[0], t, unstick=False)    # back down onto the closing curve
            return self.u

        self._check_free_flow(t, o)
        if self.mode in ALARMS:
            return self.u
        self._check_high_flow(t, o)
        if self._check_stopped(t, o):
            return self.u
        self._check_head_change(t)
        self._control(t, o)
        return self.u

    # ---- control ----------------------------------------------------------
    def _control(self, t: float, o: Order) -> None:
        if not self.rate.confident:
            fast = self._quick_rate(t)
            if not (self.cfg.physics.valve and fast and fast > 1.4 * o.target_gtt_min):
                return
            # a spike (the tube springing open): close now, don't wait for a full window
            self._settled = False
            self._search(t, (fast - o.target_gtt_min) / o.target_gtt_min)
            return
        err = (self.rate.gtt_min - o.target_gtt_min) / o.target_gtt_min
        head = self.est.state.head_inst_cm
        if abs(err) < (self.cfg.band_out if self._settled else self.cfg.band_in):
            if not self._settled:
                self._settled = self._ever_settled = True
                self.est.rebaseline()
            if self.mode == Mode.ALARM_LOW_HEAD:
                self.mode = Mode.RUNNING
            return
        if self._settled:
            self._lo = self._hi = None     # drifted out of the band: search afresh
        self._settled = False
        u_star = valve_for_flow(head, o.target_ml_min, self.cfg.physics)
        v = self.cfg.physics.valve
        if v:     # measured valve: only when it really is wide open and still short
            short = self.u <= self.cfg.safety.u_min + 1e-3 and err < -self.cfg.band_out
        else:
            short = u_star <= 0.0 and max_flow_ml_min(head, self.cfg.physics) < 0.95 * o.target_ml_min
        if short:
            p = self.cfg.physics
            need = o.target_ml_min * p.r0 + p.p_venous_cmh2o
            if self.mode != Mode.ALARM_LOW_HEAD:
                self.mode = Mode.ALARM_LOW_HEAD
                self._emit(t, "low_head", "warn",
                           f"Valve fully open but the bag is too low to reach "
                           f"{o.rate_ml_h:g} mL/h. Raise the bag about "
                           f"{max(0, need - head):.0f} cm (to ~{need:.0f} cm above the site).",
                           {"head_cm": head, "needed_cm": need})
        if v:
            self._search(t, err)
            return
        # Under-relax to avoid chasing noise in the head estimate; cap the step so
        # one bad reading can't swing the valve.
        cap = self.cfg.max_step_u
        step = max(-cap, min(cap, 0.7 * (u_star - self.u)))
        if abs(step) < 0.002:
            return      # below what the servo resolves
        self._move(self.u + step, t)

    def _quick_rate(self, t: float) -> float | None:
        """drops/min over the last 6 drops since the valve settled, or None."""
        fresh = [d for d in self.drop_log.drops if d >= self.t_move][-7:]
        if len(fresh) < 7 or t - fresh[-1] > 3.0:
            return None
        return 60.0 * 6 / (fresh[-1] - fresh[0])

    def _search(self, t: float, err: float) -> None:
        """Measured valve, by feedback rather than calibration. Flow only ever falls
        as the valve closes, but the curve has a cliff that moves with bag height
        and the tube does not come back the same way it was squeezed, so:
        - bracket: keep the last too-fast (lo) and too-slow (hi) positions and try
          halfway; until both are known, step in the error's direction, doubling;
        - every position is approached from the open side (see _move);
        - an opening move that changes nothing means the tube stayed squashed:
          release fully and allow more hysteresis from then on;
        - when the bracket is finer than the servo resolves, hold at whichever end
          was closer to target, unless both are far off (then one end is stale)."""
        v = self.cfg.physics.valve
        cap = v.max_step_u
        release = False
        if self._opened is not None:                   # judge the last opening move
            if err < 0 and self.rate.gtt_min < 1.1 * self._opened:
                release = True
                self._backlash = min(0.4, max(2 * self._backlash, 2 * cap))
                self._lo = self._hi = None             # measured on a squashed tube
            self._opened = None
        if err > 0:                                    # too fast: close (u up)
            self._lo, self._lo_err = self.u, err
            if self._hi is not None and self._hi <= self.u:
                self._hi = None
        else:
            self._hi, self._hi_err = self.u, err
            if self._lo is not None and self._lo >= self.u:
                self._lo = None
        if self._lo is not None and self._hi is not None and self._hi - self._lo < 0.2 * cap:
            best, best_err = ((self._lo, self._lo_err) if abs(self._lo_err) <= abs(self._hi_err)
                              else (self._hi, self._hi_err))
            if abs(best_err) <= 2 * self.cfg.band_out:
                self._settled = True                   # as close as this valve gets here
                if abs(self.u - best) > 1e-4:
                    self._move(best, t)
                return
            if err > 0:
                self._hi = None
            else:
                self._lo = None
        if self._lo is not None and self._hi is not None:
            self._push = 0
            u_next = (self._lo + self._hi) / 2
        else:
            self._push = self._push + 1 if self._push_dir == (err > 0) else 1
            self._push_dir = err > 0
            step = cap * min(4, 2 ** (self._push - 1))
            u_next = self.u + (step if err > 0 else -step)
        u_next = min(u_next, v.shut_u - 0.01)          # searching never shuts the tube fully
        if u_next < self.u:
            self._opened = self.rate.gtt_min           # check next time that it helped
        self._move(u_next, t, release=release)

    def _move(self, u_target: float, t: float, unstick: bool = True, release: bool = False) -> None:
        s = self.cfg.safety
        u_target = min(s.u_max, max(s.u_min, u_target))
        v = self.cfg.physics.valve
        self._unstick = None
        if unstick and v and u_target < self.u - 0.002 and (self._backlash or self.u >= v.shut_u or release):
            # opening: go past the target to free the tube, then close onto it.
            # From fully shut (or found stuck) the walls cling: release all the way.
            shut = self.u >= v.shut_u or release
            # Approach every position from the open side: back off past the target by
            # the learned hysteresis, and at least to the last angle known to flow too
            # fast (seen while closing), so the tube is always squeezed the same way.
            back = u_target - self._backlash
            if self._lo is not None and self._lo < u_target:
                back = min(back, self._lo - self._backlash / 2)
            over = s.u_min if shut else max(s.u_min, back)
            dwell = 4.0 if shut else 1.0
            self._unstick = (u_target, t + (self.u - over) / s.max_slew_per_s + dwell)
            u_target = over
        du = abs(u_target - self.u)
        if du > 1e-4:
            # Data is only fresh once the slew-limited servo has arrived and the
            # chamber has settled; drops before that reflect an in-between valve.
            self.t_move = t + du / s.max_slew_per_s + (v.settle_s if v else 1.0)
        if self._unstick:
            self.t_move = self._unstick[1]
        self.u = u_target   # slew limiting is enforced by the device layer + firmware

    # ---- diagnostics ------------------------------------------------------
    def _check_head_change(self, t: float) -> None:
        if not self.rate.confident:
            return
        s = self.est.state
        if not self._ever_settled:
            # the starting head is a guess: until the flow has been on target once,
            # a different implied head is the twin learning, not the bag moving
            s.head_baseline_cm = s.head_inst_cm
            return
        delta = s.head_inst_cm - s.head_baseline_cm
        v = self.cfg.physics.valve
        if abs(delta) > (v.head_alert_cm if v else 10.0) and not self._recent("head_change", t, 20):
            direction = "lowered" if delta < 0 else "raised"
            self._emit(t, "head_change", "warn",
                       f"Flow changed with no valve change. The twin's implied head moved "
                       f"{delta:+.0f} cm (from {s.head_baseline_cm:.0f} to {s.head_inst_cm:.0f} cm): "
                       f"the bag was probably {direction} about {abs(delta):.0f} cm, or the "
                       f"patient moved. Re-adjusting the clamp.",
                       {"delta_cm": delta, "from_cm": s.head_baseline_cm, "to_cm": s.head_inst_cm})
            s.head_baseline_cm = s.head_fast_cm = s.head_inst_cm

    def _check_stopped(self, t: float, o: Order) -> bool:
        expected_interval = 60.0 / o.target_gtt_min
        v = self.cfg.physics.valve
        stall = max(4 * expected_interval, 8.0, v.stall_s if v else 0.0)
        age = self.rate.last_drop_age_s
        if t - self.t_started < stall + 4:
            return False
        if self.mode != Mode.PROBING and t - self.t_move < stall:
            return False                  # valve just moved: give the drip time to restart
        if age < stall:
            if self._probe:
                self._emit(t, "probe_cleared", "info",
                           "Drops resumed during the flow check; no alarm.")
                u0 = self._probe["u0"]
                self.mode, self._probe = Mode.RUNNING, None
                self._move(u0, t)          # back to where control had it: no ratcheting open
            return False

        remaining = self.est.state.remaining_ml
        if remaining < max(50.0, 0.05 * o.bag_volume_ml):
            self.mode = Mode.ALARM_EMPTY
            self._move(1.0, t)
            self._emit(t, "empty_bag", "alarm",
                       f"Drops stopped with only ~{remaining:.0f} mL estimated left in the bag "
                       f"({self.est.state.infused_ml:.0f} mL infused). Bag is likely empty. "
                       f"Valve closed to keep the line from running dry.",
                       {"remaining_ml": remaining})
            return True

        if self._probe is None:
            self._say_no_flow(t, o, age)
            # Active test: open the valve a bit. A low bag would respond; a kink won't.
            self._probe = {"u0": self.u, "t0": t}
            self.mode = Mode.PROBING
            self._move(max(0.0, self.u - 0.25), t)
            self._emit(t, "probe", "warn",
                       f"No drops for {age:.0f} s with ~{remaining:.0f} mL left. "
                       f"Opening the valve slightly to tell a kink from a low bag.")
            return True

        if t - self._probe["t0"] > max(2 * expected_interval, 6.0):
            u0 = self._probe["u0"]
            self._probe = None
            self.mode = Mode.ALARM_OCCLUSION
            # Restore the pre-probe position so clearing the kink doesn't cause a bolus.
            self._move(u0, t)
            self._emit(t, "occlusion", "alarm",
                       f"OCCLUSION: no flow even with the valve opened, and ~{remaining:.0f} mL "
                       f"is still in the bag, so this is a kink or clamp, not an empty bag. "
                       f"Check the line from bag to patient and the cannula site. Valve "
                       f"returned to its previous setting to prevent a bolus when it clears.",
                       {"remaining_ml": remaining})
        return True

    def _check_high_flow(self, t: float, o: Order) -> None:
        """Flow well above target for a while: say so, with how much higher the bag
        would have to be, by the model, to give this flow at this valve setting."""
        r = self.rate
        high = (r.confident and r.gtt_min > 1.25 * o.target_gtt_min
                and self._unstick is None and t >= self.t_move and t - self.t_started > 60)
        if not high:
            self._high_since = None
            return
        if self._high_since is None:
            self._high_since = t
        if t - self._high_since < 30 or self._recent("high_flow", t, 120):
            return
        p = self.cfg.physics
        q = gtt_min_to_ml_h(r.gtt_min, o.drop_factor) / 60.0           # mL/min measured
        resistance = p.r0 + clamp_resistance(self.u, p)                 # cmH2O per mL/min
        if resistance == float("inf"):
            return
        excess_cm = (q - o.target_ml_min) * resistance / p.rho_g
        pct = (r.gtt_min / o.target_gtt_min - 1) * 100
        size = f"{excess_cm / 100:.1f} meters" if excess_cm >= 100 else f"{excess_cm:.0f} centimeters"
        size_es = f"{excess_cm / 100:.1f} metros" if excess_cm >= 100 else f"{excess_cm:.0f} centímetros"
        self._emit(t, "high_flow", "warn",
                   f"Something is wrong: flow is {r.gtt_min:.0f} drops/min, {pct:.0f}% above the "
                   f"{o.target_gtt_min:.0f} target. By the physical model the bag is about "
                   f"{excess_cm:.0f} cm higher than it should be for this valve setting. "
                   f"Lower the bag or check the tubing and the pinch valve.",
                   {"rate_gtt_min": r.gtt_min, "excess_head_cm": excess_cm,
                    "say": {"en": f"Warning. Something is wrong. Flow is {pct:.0f} percent above "
                                  f"target. By the physical model, the bag is about {size} higher "
                                  f"than it should be. Lower the bag, or check the tubing.",
                            "es": f"Atención. Algo va mal. El flujo está {pct:.0f} por ciento por "
                                  f"encima del objetivo. Según el modelo físico, la bolsa está unos "
                                  f"{size_es} más alta de lo debido. Baje la bolsa o revise el tubo."}})

    def _say_no_flow(self, t: float, o: Order, age: float) -> None:
        """Nothing dripping: if the model says the valve itself is (nearly) shut at its
        position, say so; otherwise the line must be closed somewhere else."""
        p = self.cfg.physics
        predicted = flow_ml_min(self.est.state.head_cm, self.u, p=p)
        deg = (f" at {p.valve.open_deg + (p.valve.closed_deg - p.valve.open_deg) * self.u:.0f} degrees"
               if p.valve else "")
        if predicted < 0.1 * o.target_ml_min:
            self._emit(t, "no_flow", "warn",
                       f"Nothing has dripped for {age:.0f} s: the valve is closed{deg}.",
                       {"valve_u": self.u, "say": {
                           "en": f"Nothing is dripping. The valve is closed{deg}.",
                           "es": "No está goteando. La válvula está cerrada."}})
        else:
            self._emit(t, "no_flow", "warn",
                       f"Nothing has dripped for {age:.0f} s although the valve is open{deg}: by "
                       f"the physical model the line is closed somewhere (clamp, kink or pinched "
                       f"tube).",
                       {"valve_u": self.u, "say": {
                           "en": "Nothing is dripping, although the valve is open. By the physical "
                                 "model, the line is closed somewhere: a clamp, a kink, or a "
                                 "pinched tube.",
                           "es": "No está goteando, aunque la válvula está abierta. Según el modelo "
                                 "físico, la línea está cerrada en algún punto: una pinza, un "
                                 "doblez o un tubo pinzado."}})

    def _check_free_flow(self, t: float, o: Order) -> None:
        r = self.rate.gtt_min
        hard = self.cfg.safety.hard_max_gtt_min
        # our own brief full-open to free the tube (seconds): judged once it is over
        releasing = self._unstick is not None or t < self.t_move
        # Generic model: fast with the valve near open means it isn't holding. Measured
        # valve: it starts open and closes in on the target, so "isn't holding" means
        # still far too fast with the search closed to its limit (tube out of the pinch).
        v = self.cfg.physics.valve
        failing = self.u >= v.shut_u - 0.02 if v else self.u < 0.05
        if not releasing and ((self.rate.confident and r > 1.6 * o.target_gtt_min and failing)
                              or r > hard):
            self.mode = Mode.ALARM_FREE_FLOW
            self._move(1.0, t)
            self._emit(t, "free_flow", "alarm",
                       f"FREE FLOW: {r:.0f} drops/min vs target {o.target_gtt_min:.0f}. "
                       f"Valve closed. Check that the tubing is seated in the pinch valve.",
                       {"rate_gtt_min": r})

    def _narrate(self, t: float) -> None:
        """Monitor mode has no valve, so the useful output is telling people
        what the drops are doing: started, stopped, or rate changed."""
        r = self.rate
        if not self._flowing:
            if r.confident:
                self._flowing, self._ref_rate = True, r.gtt_min
                self._emit(t, "drops_detected", "info",
                           f"Drops detected: about {r.gtt_min:.0f} drops per minute.",
                           {"rate_gtt_min": r.gtt_min})
            return
        stall = max(10.0, 3 * 60.0 / self._ref_rate)
        if r.last_drop_age_s > stall:
            self._flowing = False
            self._emit(t, "drops_stopped", "warn",
                       f"No drops for {r.last_drop_age_s:.0f} seconds. The drip has stopped, "
                       f"or the camera has lost sight of the drop.",
                       {"last_drop_age_s": r.last_drop_age_s})
        elif (r.confident and abs(r.gtt_min - self._ref_rate) > 0.2 * self._ref_rate
              and not self._recent("rate_change", t, 10)):
            self._emit(t, "rate_change", "warn",
                       f"Drip rate changed from {self._ref_rate:.0f} to {r.gtt_min:.0f} "
                       f"drops per minute.", {"from": self._ref_rate, "to": r.gtt_min})
            self._ref_rate = r.gtt_min

    # ---- helpers ----------------------------------------------------------
    def _recent(self, kind: str, t: float, within_s: float) -> bool:
        return any(e.kind == kind and t - e.t < within_s for e in self.events[-20:])

    def _emit(self, t: float, kind: str, severity: str, message: str, data: dict | None = None):
        self.events.append(Event(t, kind, severity, message, data or {}))

    def _record(self, t: float, q: float) -> None:
        self._history.append({"t": round(t, 2), "gtt_min": round(self.rate.gtt_min, 2),
                              "u": round(self.u, 3),
                              "head_cm": round(self.est.state.head_fast_cm, 1)})
        del self._history[:-600]

    def snapshot(self) -> dict:
        o, s = self.order, self.est.state
        df = o.drop_factor if o else self.cfg.monitor_drop_factor if self.monitor_only else None
        return {
            "t": self.t,
            "mode": self.mode.value,
            "monitor_only": self.monitor_only,
            "order": asdict(o) if o else None,
            "target_gtt_min": o.target_gtt_min if o else None,
            "rate_gtt_min": round(self.rate.gtt_min, 2),
            "rate_ml_h": round(gtt_min_to_ml_h(self.rate.gtt_min, df), 1) if df else None,
            "drop_volume_ml": round(drop_volume_ml(self.rate.gtt_min, df), 4) if df else None,
            "drop_factor": df,
            "rate_confident": self.rate.confident,
            "last_drop_age_s": None if math.isinf(self.rate.last_drop_age_s)
                               else round(self.rate.last_drop_age_s, 1),
            "valve_u": round(self.u, 3),
            "head_cm": round(s.head_fast_cm, 1),
            "head_baseline_cm": round(s.head_baseline_cm, 1),
            "infused_ml": round(self.monitor_ml if self.monitor_only else s.infused_ml, 1),
            "remaining_ml": round(s.remaining_ml, 1),
            "drops_counted": self.drop_log.count,
        }
