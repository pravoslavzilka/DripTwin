"""Drop log and drop-rate estimation.

Drops come from one sensor: the webcam (OpenCV, see vision.py). The detector
already debounces, so every timestamp here is one physical drop.
"""

from collections import deque
from dataclasses import dataclass
import statistics


@dataclass
class RateEstimate:
    gtt_min: float
    confident: bool
    last_drop_age_s: float
    n_intervals: int
    changepoint: float | None = None   # time a step in drop interval started


class DropLog:
    def __init__(self) -> None:
        self.drops: deque[float] = deque(maxlen=64)   # recent timestamps, for the rate
        self.count = 0                                # every drop since start

    def add(self, t: float) -> None:
        self.drops.append(t)
        self.count += 1


class RateEstimator:
    """Median of recent inter-drop intervals — robust to a single missed or
    double-counted drop. When drops stop, the 'open interval' since the last
    drop bounds the rate from above, so the estimate decays toward zero
    instead of freezing at the last good value."""

    def __init__(self, window: int = 8, mean: bool = False):
        """mean=True: for drips that come irregularly (near a pinch drops arrive
        in bursts and pairs), use the mean interval over the window. It is the
        actual flow; a median then flips between the short and long spacing."""
        self.window = window
        self.mean = mean

    def estimate(self, drops: deque[float], now: float, since: float = 0.0) -> RateEstimate:
        """`since` discards intervals from before the last valve move, so a
        fresh estimate reflects the new valve position only."""
        age = now - drops[-1] if drops else float("inf")
        fresh = [t for t in drops if t >= since][-(self.window + 1):]
        times = fresh if len(fresh) >= 3 else list(drops)[-(self.window + 1):]
        if len(times) < 3:
            return RateEstimate(0.0, False, age, max(0, len(times) - 1))
        intervals = [b - a for a, b in zip(times, times[1:])]
        if self.mean:
            avg = (times[-1] - times[0]) / len(intervals)
            on_time = age <= max(3.0 * avg, 10.0)       # pauses are normal here
            effective = avg if on_time else age
            rate = 60.0 / effective if effective > 0 else 0.0
            # a full window, or a minute of drops when the flow is slow
            enough = len(intervals) >= self.window or (len(intervals) >= 6 and times[-1] - times[0] >= 60)
            confident = times is fresh and enough and on_time
            return RateEstimate(rate, confident, age, len(intervals))
        med = statistics.median(intervals)
        # Normal jitter: trust the interval median. A gap clearly longer than
        # usual caps the rate (so it decays when drops stop) but is not trusted
        # for control - one slow interval alone is not a measurement.
        on_time = age <= 1.3 * med
        effective = med if on_time else age
        rate = 60.0 / effective if effective > 0 else 0.0
        spread = (max(intervals) - min(intervals)) / med if med > 0 else 9.9
        confident = (times is fresh and len(intervals) >= 4 and on_time and spread < 1.5)
        cp = self._changepoint(times, intervals) if times is fresh else None
        return RateEstimate(rate, confident, age, len(intervals), cp)

    @staticmethod
    def _changepoint(times: list[float], intervals: list[float]) -> float | None:
        """Three consecutive intervals all >12% off the earlier median, same
        direction, means the flow stepped; the window should restart there."""
        if len(intervals) < 6:
            return None
        base = statistics.median(intervals[:-3])
        last = intervals[-3:]
        if all(x > 1.12 * base for x in last) or all(x < 0.88 * base for x in last):
            return times[-4]
        return None
