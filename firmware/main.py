# DripTwin firmware for Raspberry Pi Pico (MicroPython).
#
# This is the safety authority. The laptop (and the AI agent behind it) can only
# *request* a valve position; everything below is enforced here and cannot be
# changed over serial:
#   - valve travel is clamped to [U_MIN, U_MAX] and slew-limited
#   - host silence > HB_TIMEOUT_MS -> hold position + alarm (acts like a manual clamp)
#   - hardware watchdog reboots a hung loop into the closed-and-alarming state
# Drops are counted by the webcam on the laptop; free flow is caught there.
#
# Wiring:
#   SG90 servo signal  -> GP15   (5 V from the power-supply module, common GND)
#   Buzzer (optional)  -> GP13
#
# Serial protocol: newline-delimited JSON over USB.
#   host -> pico  {"cmd":"set","u":0.62}  {"cmd":"hb"}  {"cmd":"close"}
#   pico -> host  {"ev":"status","u":..,"u_cmd":..,"host":..}
#                 {"ev":"alarm","kind":"host_lost"|"boot"}

import sys, json, time, select
from machine import Pin, PWM, WDT

# ---- hard limits (mirror of driptwin/config.py SafetyLimits) ----
U_MIN, U_MAX = 0.0, 1.0
MAX_SLEW_PER_S = 0.15
HB_TIMEOUT_MS = 3000
# servo calibration (python -m driptwin.valvecal): angle at u=0 (open) and
# u=1 (pinched shut). Must match open_deg/closed_deg in valve_cal.json.
OPEN_DEG, CLOSED_DEG = 40.0, 90.0
ANGLE_MIN, ANGLE_MAX = 0.0, 90.0     # mechanical range of the valve, never exceeded

servo = PWM(Pin(15)); servo.freq(50)
buzzer = Pin(13, Pin.OUT); buzzer.value(0)
led = Pin("LED", Pin.OUT)
wdt = WDT(timeout=2000)

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")


def write_servo(u):
    deg = min(ANGLE_MAX, max(ANGLE_MIN, OPEN_DEG + (CLOSED_DEG - OPEN_DEG) * u))
    servo.duty_ns(int((500 + 2000 * deg / 180) * 1000))     # SG90: 500-2500 us = 0-180 deg


# Boot closed: an unattended reboot must never open the line.
u = U_MAX
u_cmd = U_MAX
write_servo(u)
host_ok = False
last_host = time.ticks_ms()
last_servo = time.ticks_ms()
last_status = time.ticks_ms()
buf = ""

send({"ev": "alarm", "kind": "boot"})

while True:
    wdt.feed()
    now = time.ticks_ms()

    # ---- serial in (non-blocking) ----
    while poll.poll(0):
        ch = sys.stdin.read(1)
        if ch == "\n":
            try:
                msg = json.loads(buf)
                cmd = msg.get("cmd")
                last_host, host_ok = now, True
                if cmd == "set":
                    u_cmd = min(U_MAX, max(U_MIN, float(msg["u"])))
                elif cmd == "close":
                    u_cmd = U_MAX
            except Exception as e:
                send({"ev": "error", "msg": str(e)})
            buf = ""
        elif len(buf) < 200:
            buf += ch

    # ---- host watchdog: hold position, alarm ----
    if host_ok and time.ticks_diff(now, last_host) > HB_TIMEOUT_MS:
        host_ok = False
        u_cmd = u            # freeze where we are
        send({"ev": "alarm", "kind": "host_lost"})

    # ---- slew-limited servo update at 50 Hz ----
    dt = time.ticks_diff(now, last_servo)
    if dt >= 20:
        last_servo = now
        step = MAX_SLEW_PER_S * dt / 1000.0
        u += max(-step, min(step, u_cmd - u))
        write_servo(u)

    alarming = not host_ok
    buzzer.value(alarming and (now // 250) % 2)
    led.value((now // (150 if alarming else 1000)) % 2)

    if time.ticks_diff(now, last_status) >= 500:
        last_status = now
        send({"ev": "status", "u": round(u, 3), "u_cmd": round(u_cmd, 3), "host": host_ok})
