"""Voice/clinical agent on Claude.

The agent can read state, explain it, and take *orders* — but it never touches
the valve. Its tools are deliberately narrow:

  propose_order -> validates + converts mL/h to drops/min, stores as pending
  start_infusion -> only starts the order that was proposed and read back
  pause / resume / set_bag_height / get_status / log_note
  get_flow_history / find_flow_gaps / get_time_to_empty -> fixed read-only queries
      on the Tiger Data aggregates (timeseries.py), never free-form SQL

Valve position comes from the controller; hard limits live in firmware.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict
from typing import Callable

import anthropic
import httpx

from . import timeseries
from .physics import DROP_CURVES, drop_volume_ml
from .twin import DripTwin, Order, OrderRejected, validate_order

MODEL = "claude-opus-5-5"

# Words that count as the person confirming a read-back order (English and Spanish).
CONFIRM = re.compile(r"\b(confirm(ed)?|yes|yeah|yep|correct|affirmative|go ahead|"
                     r"confirmo|confirmado|s[ií]|correcto|adelante)\b", re.I)
LANG_NAMES = {"es": "Spanish"}

SYSTEM = """You are DripTwin, a voice assistant clipped onto a gravity IV line. You talk \
with nurses, field medics and vets through a speaker, so every reply is spoken aloud.

How to speak:
- Answer only what was asked, in as few words as possible: "How many drops so far?" gets \
"Sixteen drops so far." Don't add the rate, the mode, or caveats unless asked. Use up to \
three short sentences only for alarms or orders. No markdown, lists, or symbols; write \
numbers so they read naturally ("forty drops per minute").
- For alarms, lead with what matters for the patient, then the reason.
- A "[reply in Spanish]" note means the person is speaking Spanish: answer in natural, \
plain Spanish, numbers in words ("veintiséis gotas por minuto"). Device data stays in English; \
translate it.

What you can do:
- Take infusion orders. Always call propose_order first, read the result back (rate, set, \
drops per minute, volume), and ask the person to say "confirm". Call start_infusion only \
when their next message confirms that exact order; the device rejects a start in the same \
turn as the proposal or without a spoken confirmation. If an order is missing the drop \
factor or volume, ask.
- Explain what the line is doing using get_status. The status includes a physics "digital \
twin": the implied head (bag height above the site) is back-calculated from drop rate and \
valve position. A step change in implied head with no valve change means the bag or patient \
moved. No drops with plenty of fluid left, even after the valve opened, means a kink or \
occlusion. No drops with the bag nearly empty means an empty bag.
- Pause, resume, record the bag height, and log notes.
- For the past and the future ("what happened in the last ten minutes", "when will the \nbag run out", "did it stop earlier"), use get_flow_history, find_flow_gaps and \nget_time_to_empty. They read the infusion log; use get_status for the present.
- If get_status shows monitor_only, no valve is connected: the webcam is only counting \
drops. Answer count and rate questions from it. Mention that infusions can't be started \
only when someone asks to start or change one.

Hard rules:
- You cannot set the valve directly and must not claim to. The controller sets it and the \
firmware enforces safety limits you cannot change.
- Never recommend drugs or doses. You run the order you are given, within device limits.
- For any alarm, say what to physically check. If unsure, tell them to check the patient \
and the line by hand.
- A turn may start with [reply in Spanish]: then answer in Spanish, whatever language the \
question was in. Otherwise answer in English.
- Device events and a live status snapshot arrive in brackets at the start of a user turn; \
they come from the device, not the person. The snapshot is current as of the question, so \
answer from it directly; call get_status only to re-check after you change something."""


TOOLS = [
    {
        "name": "propose_order",
        "description": "Validate an infusion order and convert it to drops per minute. Does not "
                       "start anything. Use whenever the person states or changes an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rate_ml_h": {"type": "number", "description": "Ordered rate in mL per hour"},
                "drop_factor": {"type": "integer", "description": "Set calibration in drops/mL: 10, 15, 20 or 60"},
                "vtbi_ml": {"type": "number", "description": "Volume to be infused in mL"},
                "bag_volume_ml": {"type": "number", "description": "Volume currently in the bag, mL"},
                "drug": {"type": "string", "description": "Fluid or drug name as stated"},
            },
            "required": ["rate_ml_h", "drop_factor", "vtbi_ml", "bag_volume_ml", "drug"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "start_infusion",
        "description": "Start the pending order from propose_order. Only call after the person "
                       "explicitly confirmed the read-back.",
        "input_schema": {
            "type": "object",
            "properties": {"confirmation_quote": {
                "type": "string", "description": "The person's words confirming the order"}},
            "required": ["confirmation_quote"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_status",
        "description": "Current rate, target, valve, digital-twin head estimate, volume infused "
                       "and remaining, mode, and recent device events with explanations.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "pause_infusion",
        "description": "Close the valve and pause the infusion.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "resume_infusion",
        "description": "Resume after a pause or after an alarm the person says they fixed.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "set_bag_height",
        "description": "Record the bag height above the IV site, as measured by the person.",
        "input_schema": {
            "type": "object",
            "properties": {"height_cm": {"type": "number"}},
            "required": ["height_cm"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "log_note",
        "description": "Add a note to the infusion log (observations, site checks, handover).",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
TOOLS += timeseries.TOOLS


class Agent:
    NAME = "Claude"

    def __init__(self, twin: DripTwin, clock: Callable[[], float] = time.monotonic,
                 on_log: Callable[[str, dict], None] | None = None,
                 lock: threading.Lock | None = None):
        self.twin = twin
        self.lock = lock or threading.Lock()
        self.clock = clock
        self.on_log = on_log or (lambda kind, data: None)
        self.client = self._connect()
        self.messages: list[dict] = []
        self.pending: Order | None = None
        self._events_seen = 0
        self._turn = 0              # one per question; read-back and confirm must be different turns
        self._pending_turn = -1
        self._last_text = ""        # the person's own words this turn, not the model's paraphrase

    def _connect(self):
        return anthropic.Anthropic()

    # ---- tools -------------------------------------------------------------
    def _tool(self, name: str, args: dict) -> tuple[str, bool]:
        if name in timeseries.TOOL_NAMES:   # database round trip: don't hold the twin's lock
            return timeseries.run_tool(name, args)
        with self.lock:   # the control loop ticks the twin from another thread
            return self._tool_locked(name, args)

    def _tool_locked(self, name: str, args: dict) -> tuple[str, bool]:
        t = self.clock()
        if name == "propose_order":
            try:
                order = Order(float(args["rate_ml_h"]), int(args["drop_factor"]),
                              float(args["vtbi_ml"]), float(args["bag_volume_ml"]),
                              args.get("drug") or "fluid")
                validate_order(order)
                if self.twin.monitor_only:
                    raise OrderRejected("monitor-only mode, no valve connected")
            except (OrderRejected, ValueError, KeyError) as e:
                return f"Rejected by device limits: {e}", True
            self.pending, self._pending_turn = order, self._turn
            hours = order.vtbi_ml / order.rate_ml_h
            return json.dumps({**asdict(order),
                               "drops_per_min": round(order.target_gtt_min, 1),
                               "ml_per_drop": round(drop_volume_ml(order.target_gtt_min,
                                                                   order.drop_factor), 4),
                               "drop_size": ("measured for this set" if order.drop_factor
                                             in DROP_CURVES else "nominal 1/drop factor"),
                               "seconds_per_drop": round(60 / order.target_gtt_min, 2),
                               "duration_h": round(hours, 2),
                               "status": "pending - read back and get confirmation"}), False
        if name == "start_infusion":
            if self.pending is None:
                return "No pending order. Call propose_order first.", True
            # Enforced here, not only in the prompt: no start without a spoken "confirm"
            # that came after the read-back.
            if self._pending_turn == self._turn:
                return ("Not started: read the order back and wait for the person to confirm "
                        "in their next message."), True
            if not CONFIRM.search(self._last_text):
                return ("Not started: the person hasn't confirmed. Read the order back and ask "
                        "them to say confirm."), True
            try:
                self.twin.start(self.pending, t)
            except OrderRejected as e:
                return f"Rejected by device: {e}", True
            self.on_log("order", {**asdict(self.pending),
                                  "confirmation": args.get("confirmation_quote", "")})
            self.pending = None
            return "Started. " + self.twin.events[-1].message, False
        if name == "get_status":
            return self._status(t), False
        if name == "pause_infusion":
            self.twin.pause(t)
            return self.twin.snapshot()["mode"], False
        if name == "resume_infusion":
            self.twin.resume(t)
            return self.twin.snapshot()["mode"], False
        if name == "set_bag_height":
            h = float(args["height_cm"])
            if not 20 <= h <= 250:
                return "Height must be between 20 and 250 cm.", True
            self.twin.set_head_hint(h, t)
            return f"Recorded {h:.0f} cm.", False
        if name == "log_note":
            self.on_log("note", {"note": args["note"]})
            return "Logged.", False
        return f"Unknown tool {name}", True

    def _status(self, t: float) -> str:
        snap = self.twin.snapshot()
        snap["recent_events"] = [
            {"seconds_ago": round(t - e.t), "kind": e.kind, "severity": e.severity,
             "message": e.message} for e in self.twin.events[-6:]]
        return json.dumps(snap)

    # ---- conversation --------------------------------------------------------
    def _device_context(self) -> str:
        new = self.twin.events[self._events_seen:]
        self._events_seen = len(self.twin.events)
        warn = [e for e in new if e.severity != "info"]
        if not warn:
            return ""
        return "[device events: " + " | ".join(f"{e.kind}: {e.message}" for e in warn[-3:]) + "]\n"

    def _turn_context(self, lang: str = "en") -> str:
        # Live status rides along with the question, so most answers need no
        # get_status round trip: one model call instead of two.
        reply_in = f"[reply in {LANG_NAMES[lang]}]\n" if lang in LANG_NAMES else ""
        return (reply_in + self._device_context()
                + f"[live status: {self._status(self.clock())}]\n")

    def _begin(self, text: str, lang: str) -> str:
        with self.lock:
            self._turn += 1
            self._last_text = text
            return self._turn_context(lang)

    def ask(self, text: str, on_text: Callable[[str], None] | None = None,
            lang: str = "en") -> str:
        """Answer one question. on_text gets the reply as it's written, for streaming
        speech; the return value is the same text in full."""
        ctx = self._begin(text, lang)
        said: list[str] = []

        def emit(t: str) -> None:
            said.append(t)
            if on_text:
                on_text(t)
        start = len(self.messages)
        self.messages.append({"role": "user", "content": ctx + text})
        try:
            for _ in range(6):
                if said and not said[-1].endswith((" ", "\n")):
                    emit(" ")                          # text before and after a tool call
                with self.client.beta.messages.stream(
                    model=MODEL,
                    max_tokens=8000,                   # thinking counts toward this
                    system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOLS,
                    messages=self.messages,
                    output_config={"effort": "low"},   # spoken replies: favour latency
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    for delta in stream.text_stream:
                        emit(delta)
                    resp = stream.get_final_message()
                if resp.stop_reason == "refusal":
                    # drop the whole turn: a tool_use left without its result breaks every later request
                    del self.messages[start:]
                    emit(" I can't help with that one. Please check the line and the patient directly.")
                    return "".join(said).strip()
                self.messages.append({"role": "assistant", "content": resp.content})
                calls = [b for b in resp.content if b.type == "tool_use"]
                if resp.stop_reason != "tool_use" or not calls:
                    reply = "".join(said).strip()
                    if not reply:
                        reply = "Sorry, I didn't get an answer out. Could you ask again?"
                        emit(reply)
                    return reply
                results = []
                for c in calls:
                    out, err = self._tool(c.name, c.input)
                    results.append({"type": "tool_result", "tool_use_id": c.id,
                                    "content": out, "is_error": err})
                self.messages.append({"role": "user", "content": results})
        except Exception:
            del self.messages[start:]     # don't leave a half-finished turn in the history
            raise
        emit(" Sorry, I lost track there. Could you say that again?")
        return "".join(said).strip()


class GeminiAgent(Agent):
    """Fallback when only a Gemini key is set, through Google's OpenAI-compatible
    endpoint. Same system prompt, tools and safety rules; Claude stays primary."""

    NAME = "Gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    # Gemini's function declarations don't accept these schema keys.
    TOOLS = [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": {k: v for k, v in t["input_schema"].items() if k != "additionalProperties"}}}
        for t in TOOLS]

    def _connect(self):
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        return httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=30)

    def ask(self, text: str, on_text: Callable[[str], None] | None = None,
            lang: str = "en") -> str:
        reply = self._ask(self._begin(text, lang), text)
        if on_text:
            on_text(reply)                        # no streaming here: speak it in one go
        return reply

    def _ask(self, ctx: str, text: str) -> str:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": ctx + text})
        try:
            for _ in range(6):
                r = self.client.post(self.URL, json={
                    "model": self.model, "tools": self.TOOLS,
                    "messages": [{"role": "system", "content": SYSTEM}, *self.messages]})
                if r.status_code >= 400:
                    raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
                msg = r.json()["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                # tool_calls are kept whole: they can carry thought signatures Gemini needs back
                self.messages.append({"role": "assistant", "content": msg.get("content") or "",
                                      **({"tool_calls": calls} if calls else {})})
                if not calls:
                    return (msg.get("content") or "").strip()
                for c in calls:
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    out, err = self._tool(c["function"]["name"], args)
                    self.messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                          "content": ("Error: " if err else "") + out})
        except Exception:
            del self.messages[start:]     # don't leave a half-finished turn in the history
            raise
        return "Sorry, I lost track there. Could you say that again?"


def offline_explain(twin: DripTwin) -> str:
    """Spoken status without the LLM — used when there is no network."""
    s = twin.snapshot()
    if s["monitor_only"]:
        age = s["last_drop_age_s"]
        return (f"Monitoring only. Counting {s['rate_gtt_min']:.0f} drops per minute, "
                f"{s['drops_counted']} drops seen"
                + (f", last one {age:.0f} seconds ago." if age is not None else "."))
    if not s["order"]:
        return "No infusion running."
    last = next((e for e in reversed(twin.events) if e.severity != "info"), None)
    base = (f"Running at {s['rate_gtt_min']:.0f} drops per minute, target "
            f"{s['target_gtt_min']:.0f}. {s['infused_ml']:.0f} millilitres infused.")
    return (last.message + " " + base) if last else base

