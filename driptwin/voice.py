"""Cartesia speech.

  Ink 2         hands-free speech-to-text with built-in turn detection. The browser
                streams the microphone straight to Cartesia with a short-lived
                access token from access_token(), so audio never round-trips here.
  Ink-Whisper   push-to-talk fallback: one recorded clip, transcribed in one call.
  Sonic         text-to-speech, streamed over one persistent WebSocket so the first
                audio arrives while Claude is still writing (and alarms start fast).

Everything spoken goes through normalize() and, in English, a pronunciation
dictionary: "120 mL/h" must never be heard as "120 M L H".
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import AsyncIterable, AsyncIterator

import httpx

API = "https://api.cartesia.ai"
WS_API = "wss://api.cartesia.ai"
VERSION = "2026-08-14"
TTS_MODEL = os.environ.get("CARTESIA_TTS_MODEL", "sonic-3.6")
VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
SAMPLE_RATE = 24000                 # streamed PCM, s16le mono; the browser plays it as is
LANGS = ("en", "es")

# Calm for answers and status, urgent for alarms. Sonic only honours `emotion`
# in English, so it's dropped for other languages. Sonic infers tone from the
# text when no emotion is given, which suits the alarm wording; set
# CARTESIA_ALARM_EMOTION to force one (neutral, calm, angry, content, sad).
STYLES = {
    "calm": {"emotion": "calm", "speed": 0.95, "volume": 1.0},
    "alarm": {"speed": 1.15, "volume": 1.6,
              **({"emotion": e} if (e := os.environ.get("CARTESIA_ALARM_EMOTION")) else {})},
}

# Words Ink 2 should expect: boosts recognition of units and drug names in orders.
KEYTERMS = ["Assistant", "mL per hour", "milliliters per hour", "drop set", "drops per minute",
            "gtt", "normal saline", "lactated Ringer's", "D5W", "heparin", "vancomycin",
            "ceftriaxone", "metronidazole", "oxytocin", "confirm"]


def available() -> bool:
    return bool(os.environ.get("CARTESIA_API_KEY"))


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['CARTESIA_API_KEY']}",
            "Cartesia-Version": VERSION}


# ---- what gets said ---------------------------------------------------------
# Units are rewritten here rather than in the dictionary: "mL/h" isn't a word a
# dictionary can match, and alarms skip Claude, so nothing else spells them out.
_UNITS = {
    "en": [(r"\b(?:gtts?|drops)\s*/\s*min(?:ute)?\b", "drops per minute"),
           (r"\bm[lL]\s*/\s*h(?:ou)?r?\b", "milliliters per hour"),
           (r"\bm[lL]\s*/\s*min\b", "milliliters per minute"),
           (r"\bgtts?\b", "drops"),
           (r"\bm[lL]\b", "milliliters"),
           (r"\bcm\b", "centimeters"),
           (r"~\s*", "about "),
           (r"(\d)\s*%", r"\1 percent"),
           (r"\b(1|one) (milliliter|drop|centimeter)s\b", r"\1 \2")],
    "es": [(r"\b(?:gtts?|gotas)\s*/\s*min(?:uto)?\b", "gotas por minuto"),
           (r"\bm[lL]\s*/\s*h(?:ora)?\b", "mililitros por hora"),
           (r"\bm[lL]\s*/\s*min\b", "mililitros por minuto"),
           (r"\bgtts?\b", "gotas"),
           (r"\bm[lL]\b", "mililitros"),
           (r"\bcm\b", "centímetros"),
           (r"~\s*", "aproximadamente "),
           (r"(\d)\s*%", r"\1 por ciento"),
           (r"\b(1|un|una) (mililitro|gota|centímetro)s\b", r"\1 \2")],
}


def normalize(text: str, lang: str = "en") -> str:
    for pat, rep in _UNITS.get(lang, _UNITS["en"]):
        text = re.sub(pat, rep, text)
    return text


# Sounds-like spellings for drug names and abbreviations (English voice only).
PRONUNCIATIONS = [
    {"text": "heparin", "pronunciation": "HEP-uh-rin"},
    {"text": "vancomycin", "pronunciation": "van-koh-MY-sin"},
    {"text": "ceftriaxone", "pronunciation": "sef-try-AK-sohn"},
    {"text": "metronidazole", "pronunciation": "meh-troh-NYE-duh-zohl"},
    {"text": "piperacillin", "pronunciation": "pih-PER-uh-sil-in"},
    {"text": "tazobactam", "pronunciation": "tay-zoh-BAK-tam"},
    {"text": "oxytocin", "pronunciation": "ok-see-TOH-sin"},
    {"text": "furosemide", "pronunciation": "fyoo-ROH-seh-mide"},
    {"text": "ondansetron", "pronunciation": "on-DAN-seh-tron"},
    {"text": "acetaminophen", "pronunciation": "uh-see-tuh-MIN-uh-fen"},
    {"text": "NS", "pronunciation": "normal saline", "case_sensitive": True},
    {"text": "LR", "pronunciation": "lactated Ringer's", "case_sensitive": True},
    {"text": "D5W", "pronunciation": "D five W", "case_sensitive": True},
    {"text": "KCl", "pronunciation": "potassium chloride", "case_sensitive": True},
]
DICT_CACHE = Path(__file__).resolve().parent.parent / ".cartesia_dict.json"
_dict_id: str | None = None


async def ensure_dictionary() -> str | None:
    """Create the pronunciation dictionary once and reuse it across restarts. The
    cache is keyed on the entries, so editing PRONUNCIATIONS makes a new one."""
    global _dict_id
    if env := os.environ.get("CARTESIA_PRONUNCIATION_DICT_ID"):
        _dict_id = env
        return env
    digest = hashlib.sha256(json.dumps(PRONUNCIATIONS, sort_keys=True).encode()).hexdigest()[:16]
    try:
        cached = json.loads(DICT_CACHE.read_text())
        if cached.get("hash") == digest:
            _dict_id = cached["id"]
            return _dict_id
    except (OSError, ValueError, KeyError):
        pass
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{API}/pronunciation-dicts/", headers=_headers(), json={
            "name": f"DripTwin clinical terms {digest[:6]}", "items": PRONUNCIATIONS})
        r.raise_for_status()
        _dict_id = r.json()["id"]
    DICT_CACHE.write_text(json.dumps({"hash": digest, "id": _dict_id}))
    return _dict_id


def _request(lang: str, style: str) -> dict:
    cfg = dict(STYLES[style])
    if lang != "en":
        cfg.pop("emotion", None)
    req = {"model_id": TTS_MODEL, "voice": {"id": VOICE_ID}, "language": lang,
           "generation_config": cfg}
    if lang == "en" and _dict_id:
        req["pronunciation_dict_id"] = _dict_id
    return req


# ---- speech-to-text -----------------------------------------------------------
async def access_token(seconds: int = 600) -> str:
    """Short-lived token so the browser can open the Ink 2 socket without the API key."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{API}/access-token", headers=_headers(),
                         json={"grants": {"stt": True}, "expires_in": seconds})
        r.raise_for_status()
        return r.json()["token"]


async def transcribe(audio: bytes, filename: str = "speech.webm", lang: str = "en") -> str:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{API}/stt", headers=_headers(),
                         files={"file": (filename, audio)},
                         data={"model": "ink-whisper", "language": lang})
        r.raise_for_status()
        return r.json()["text"].strip()


# ---- text-to-speech -------------------------------------------------------------
async def speak(text: str, lang: str = "en", style: str = "calm") -> bytes:
    """Whole reply as one MP3: the fallback when the streaming socket fails."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{API}/tts/bytes", headers=_headers(), json={
            **_request(lang, style),
            "transcript": normalize(text, lang),
            "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000},
        })
        r.raise_for_status()
        return r.content


class VoiceError(RuntimeError):
    pass


class _Socket:
    """One TTS WebSocket shared by every reply and alarm, multiplexed by context_id.
    Opening a socket costs a TLS handshake, which an alarm shouldn't wait for."""

    def __init__(self):
        self.ws = None
        self.contexts: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def get(self):
        async with self._lock:
            if self.ws is None:
                import websockets
                url = f"{WS_API}/tts/websocket?cartesia_version={VERSION}"
                self.ws = await websockets.connect(url, additional_headers={
                    "X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": VERSION},
                    max_size=None)
                asyncio.create_task(self._read(self.ws))
            return self.ws

    async def _read(self, ws) -> None:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if q := self.contexts.get(msg.get("context_id")):
                    q.put_nowait(msg)
        except Exception as e:
            print("[tts] socket closed:", repr(e))
        finally:
            if self.ws is ws:
                self.ws = None
            for q in self.contexts.values():
                q.put_nowait({"type": "error", "message": "TTS connection closed"})


_sock = _Socket()


async def warm_up() -> None:
    """Open the TTS socket and create the dictionary at startup, not on the first alarm."""
    for step in (ensure_dictionary, _sock.get):
        try:
            await step()
        except Exception as e:
            print(f"[tts] {step.__name__} failed: {e!r}")


async def _speakable(parts: AsyncIterable[str], lang: str) -> AsyncIterator[str]:
    # Normalize whole words only: "mL/h" may arrive as "mL" then "/h".
    buf = ""
    async for part in parts:
        buf += part
        cut = max(buf.rfind(" "), buf.rfind("\n")) + 1
        if cut:
            yield normalize(buf[:cut], lang)
            buf = buf[cut:]
    if buf:
        yield normalize(buf, lang)


async def _one(text: str) -> AsyncIterator[str]:
    yield text


async def stream_speech(parts: AsyncIterable[str] | str, lang: str = "en",
                        style: str = "calm") -> AsyncIterator[bytes]:
    """Yield PCM (s16le, SAMPLE_RATE) as Sonic produces it. `parts` can be a live
    stream of text deltas: each is sent as a continuation of the same context."""
    if isinstance(parts, str):
        parts = _one(parts)
    ctx, q = uuid.uuid4().hex, asyncio.Queue()
    base = {**_request(lang, style), "context_id": ctx, "max_buffer_delay_ms": 500,
            "output_format": {"container": "raw", "encoding": "pcm_s16le",
                              "sample_rate": SAMPLE_RATE}}
    ws = await _sock.get()
    _sock.contexts[ctx] = q
    finished = False

    async def feed():
        try:
            async for text in _speakable(parts, lang):
                if text.strip():
                    await ws.send(json.dumps({**base, "transcript": text, "continue": True}))
            await ws.send(json.dumps({**base, "transcript": "", "continue": False}))
        except Exception as e:
            q.put_nowait({"type": "error", "message": f"feeding text failed: {e!r}"})

    feeder = asyncio.create_task(feed())
    try:
        while True:
            msg = await q.get()
            kind = msg.get("type")
            if kind == "chunk" and msg.get("data"):
                yield base64.b64decode(msg["data"])
            elif kind == "error":
                raise VoiceError(msg.get("message") or msg.get("title") or "TTS error")
            if kind == "done" or msg.get("done"):
                finished = True
                return
    finally:
        feeder.cancel()
        _sock.contexts.pop(ctx, None)
        if not finished and _sock.ws is ws:
            try:
                await ws.send(json.dumps({"context_id": ctx, "cancel": True}))
            except Exception:
                pass


# ---- alarms -----------------------------------------------------------------------
# Short fixed lines, spoken straight from the device event without waiting on the
# LLM. The full explanation stays on the dashboard and with the assistant.
ALARM_SPEECH = {
    "occlusion": {
        "en": "Alarm. Line blocked. Check the tubing from the bag to the patient, and the IV site.",
        "es": "Alarma. Línea bloqueada. Revise el tubo desde la bolsa hasta el paciente, y el sitio de la vía.",
    },
    "empty_bag": {
        "en": "Alarm. The bag is empty. The valve is closed. Replace the bag.",
        "es": "Alarma. La bolsa está vacía. La válvula está cerrada. Cambie la bolsa.",
    },
    "free_flow": {
        "en": "Alarm. Free flow. The valve is closed. Check the tubing is seated in the pinch valve.",
        "es": "Alarma. Flujo libre. La válvula está cerrada. Revise que el tubo esté bien colocado en la válvula.",
    },
    "fw_host_lost": {
        "en": "Alarm. The clamp controller lost connection. It is holding its position.",
        "es": "Alarma. El controlador perdió la conexión. Mantiene la válvula en su posición.",
    },
}
ALARM_FALLBACK = {"en": "Alarm. Check the line and the patient.",
                  "es": "Alarma. Revise la línea y al paciente."}


def alarm_lines(kind: str, setting: str, say: dict | None = None) -> list[tuple[str, str]]:
    """(text, lang) to speak for an alarm. setting: "en", "es" or "both". `say` is an
    event's own words ({"en": ..., "es": ...}), used instead of the fixed line."""
    langs = ("en", "es") if setting == "both" else (setting if setting in LANGS else "en",)
    lines = {**ALARM_FALLBACK, **say} if say else ALARM_SPEECH.get(kind, ALARM_FALLBACK)
    return [(lines[lang], lang) for lang in langs]
