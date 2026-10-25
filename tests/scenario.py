"""Run the twin against the simulated plant. Shared by tests and the CLI."""
from driptwin.sim import SimPlant
from driptwin.twin import DripTwin, Order

def run(seconds, disturbances=(), order=None, plant=None, dt=0.1, tick=0.5, twin=None):
    plant = plant or SimPlant()
    twin = twin or DripTwin(head_cm=80.0)
    order = order or Order(120, 20, 1000, 1000)
    twin.start(order, 0.0)
    plant.command(twin.u)
    log, t, next_tick = [], 0.0, tick
    dist = sorted(disturbances, key=lambda d: d[0])
    while t < seconds:
        t = round(t + dt, 6)
        while dist and dist[0][0] <= t:
            dist.pop(0)[1](plant)
        for d in plant.step(t, dt):
            twin.on_drop(d)
        if t >= next_tick:
            next_tick += tick
            plant.command(twin.tick(t))
            log.append((t, twin.rate.gtt_min, twin.u, plant.u, twin.mode.value, plant.head_cm))
    return twin, plant, log
