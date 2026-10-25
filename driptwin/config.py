"""Central configuration. Everything safety-relevant lives in SAFETY and is
mirrored in firmware/main.py — firmware is the authority, the host copy only
lets us refuse bad commands early and explain why."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SafetyLimits:
    # Order limits (what the agent / UI may request)
    min_rate_ml_h: float = 5.0
    max_rate_ml_h: float = 500.0
    allowed_drop_factors: tuple = (10, 15, 20, 60)
    max_volume_ml: float = 3000.0

    # Valve: u = 0 fully open, 1 fully closed (pinched)
    u_min: float = 0.0
    u_max: float = 1.0
    max_slew_per_s: float = 0.15       # max valve travel per second
    # Free-flow guard: the twin closes the valve above this camera-counted drop rate
    hard_max_gtt_min: float = 180.0
    # Firmware watchdog: no heartbeat for this long -> hold + alarm
    heartbeat_timeout_s: float = 3.0


@dataclass(frozen=True)
class ValveTable:
    """The pinch valve as measured on the rig (python -m driptwin.valvecal):
    flow at each servo position, taken at one bag height. u is the servo
    position, 0 = open_deg .. 1 = closed_deg, linear in angle."""
    u: tuple                    # increasing, 0..1
    q_ml_min: tuple             # flow at each u, at head_cm
    head_cm: float
    open_deg: float
    closed_deg: float
    # The tube sticks once squashed: reopening needs this much more travel than
    # closing did, so opening moves overshoot by it and settle back while closing.
    backlash_u: float = 0.0
    # Squashed fully shut, it reopens only near u=0 (the release angle).
    shut_u: float = 0.99
    # Seconds after the servo arrives before drops reflect the new position: the
    # drip chamber's air pocket buffers flow changes.
    settle_s: float = 1.0
    # Drops through this valve come irregularly: rate = mean interval over this
    # many drops (0 = the usual median estimate).
    rate_window: int = 0
    # A gap this long (s) counts as "drops stopped"; irregular drips pause on their own.
    stall_s: float = 0.0
    # Implied-head jump (cm) reported as "bag moved"; noisy drips need more.
    head_alert_cm: float = 10.0

    step_deg: float = 1.0                # largest correction, degrees

    @property
    def max_step_u(self) -> float:
        return self.step_deg / abs(self.closed_deg - self.open_deg)


@dataclass(frozen=True)
class PhysicsParams:
    """Lumped hydraulic model of a gravity set.

    Q [mL/min] = (H - P_ven) / (R0 + Rc(u))
    H      head: bag fluid surface above the cannula [cmH2O]
    P_ven  venous back-pressure [cmH2O]
    R0     tubing + cannula resistance with the clamp wide open [cmH2O/(mL/min)]
    Rc(u)  pinch-valve resistance; slot flow ~ 1/gap^3, gap = 1 - u
    """
    rho_g: float = 1.0                   # cmH2O per cm of water column
    p_venous_cmh2o: float = 8.0
    r0: float = 5.0
    kc: float = 1.0
    u_occlude: float = 0.97              # beyond this the tube is sealed
    bag_height_cm_per_l: float = 15.0    # fluid surface drop as a 1 L bag drains
    valve: ValveTable | None = None      # measured valve; replaces the 1/gap^3 curve


@dataclass(frozen=True)
class Config:
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    physics: PhysicsParams = field(default_factory=PhysicsParams)
    control_period_s: float = 0.5
    rate_window_drops: int = 8           # inter-drop intervals in the rate estimate
    # Tolerance band with hysteresis: "in range" once within band_in of the target,
    # corrections resume only beyond band_out (drop-rate noise is a few %).
    band_in: float = 0.05
    band_out: float = 0.08
    max_step_u: float = 0.1              # one correction moves the valve at most this far
    default_head_cm: float = 80.0
    # Monitor mode has no order to say which set is on: flow is shown for this one.
    monitor_drop_factor: int = 20


CONFIG = Config()
