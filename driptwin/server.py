"""FastAPI server: live dashboard, voice endpoint, demo disturbance buttons.

    python -m driptwin.server --sim                      # no hardware
    python -m driptwin.server --port /dev/tty.usbmodem101 --camera 0
    python -m driptwin.server --monitor --camera 0       # webcam drop count only, no valve
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import voice
from .agent import Agent, GeminiAgent, offline_explain
from .runtime import Runtime
from .twin import Order, OrderRejected

WEB = Path(__file__).resolve().parent.parent / "web"
ENV_FILE = WEB.parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines from .env into os.environ. Variables already set in
    the shell win, and empty values are skipped so they don't mask a real key."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        key, val = key.strip(), val.strip().strip("'\"")
        if val and not os.environ.get(key):
            os.environ[key] = val
ARGS = argparse.Namespace(sim=True, port=None, camera=0, head=None, monitor=False)
rt: Runtime
agent: Agent | None = None
clients: set[asyncio.Queue] = set()      # open dashboards: alarm audio goes to all of them
# Voice language from the dashboard: "en", "es", or "both" (alarms in English then
# Spanish, answers in English).
VOICE = {"lang": "en"}


def reply_lang(setting: str) -> str:
    return "es" if setting == "es" else "en"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rt, agent
    rt = Runtime(sim=ARGS.sim, port=ARGS.port, camera=ARGS.camera, head_cm=ARGS.head,
                 monitor=ARGS.monitor)
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        agent = Agent(rt.twin, clock=rt.now, on_log=rt.log, lock=rt.lock)
    elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        agent = GeminiAgent(rt.twin, clock=rt.now, on_log=rt.log, lock=rt.lock)
    print(f"[agent] {agent.NAME if agent else 'none (no API key): scripted status replies'}")
    tasks = [asyncio.create_task(rt.run()), asyncio.create_task(alarm_speaker())]
    if voice.available():
        tasks.append(asyncio.create_task(voice.warm_up()))
    yield
    for t in tasks:
        t.cancel()
    if hasattr(rt.device, "close"):
        rt.device.close()          # real valve: opened, firmware stopped


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")     # dashboard CSS


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q: asyncio.Queue = asyncio.Queue()
    rt.subscribers.add(q)
    clients.add(q)
    try:
        await sock.send_json({"type": "hello", "agent": agent is not None,
                              "agent_name": agent.NAME if agent else "Claude",
                              "voice": voice.available(), "store": rt.store.kind,
                              "sim": rt.sim, "monitor": rt.monitor, "lang": VOICE["lang"]})
        while True:
            await sock.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        rt.subscribers.discard(q)
        clients.discard(q)


@app.get("/video.mjpg")
async def video():
    async def gen():
        last = None
        while True:
            jpg = rt.device.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1 / 20)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/frame.jpg")
def frame():
    """Latest annotated frame. The dashboard polls this rather than the MJPEG
    stream, which some browsers (Safari) stop updating after a few seconds."""
    jpg = rt.device.jpeg
    if jpg is None:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class Say(BaseModel):
    text: str
    speak: bool = True
    wake: bool = False      # hands-free: only answer if it starts with the wake word
    lang: str | None = None


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _audio(pcm: bytes, **extra) -> dict:
    return {"type": "audio", "pcm": base64.b64encode(pcm).decode(),
            "rate": voice.SAMPLE_RATE, **extra}


async def _answer(text: str, speak: bool, lang: str):
    """NDJSON lines: text deltas and PCM audio as they're produced, then "done".
    Claude's words go into Sonic as they're written, so speech starts before the
    reply is finished. If streaming speech fails, the whole reply is sent as one
    MP3; if that fails too, the browser speaks it with its own voice."""
    loop = asyncio.get_running_loop()
    deltas: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()
    ended = False

    def on_text(t: str) -> None:            # called from the agent's worker thread
        loop.call_soon_threadsafe(deltas.put_nowait, t)

    async def run_agent() -> str:
        try:
            if agent is None:
                reply = offline_explain(rt.twin)
                on_text(reply)
                return reply
            return await asyncio.to_thread(agent.ask, text, on_text, lang)
        except Exception as e:      # network / API failure must not break the device
            print("[agent]", repr(e))
            reply = "I can't reach the assistant right now. " + offline_explain(rt.twin)
            on_text(reply)
            return reply
        finally:
            loop.call_soon_threadsafe(deltas.put_nowait, None)

    async def texts():
        nonlocal ended
        while not ended:
            t = await deltas.get()
            if t is None:
                ended = True
                return
            out.put_nowait({"type": "text", "text": t})
            yield t

    async def speech() -> None:
        spoken = False
        try:
            if speak and voice.available():
                async for pcm in voice.stream_speech(texts(), lang):
                    spoken = True
                    out.put_nowait(_audio(pcm))
        except Exception as e:
            print("[tts] streaming failed:", repr(e))
        async for _ in texts():             # no voice, or it failed: still stream the text
            pass
        reply = await agent_task
        if speak and voice.available() and not spoken:
            try:
                mp3 = await voice.speak(reply, lang)
                out.put_nowait({"type": "mp3", "audio_b64": base64.b64encode(mp3).decode()})
                spoken = True
            except Exception as e:
                print("[tts]", repr(e))
        out.put_nowait({"type": "done", "reply": reply, "spoken": spoken})
        out.put_nowait(None)

    agent_task = asyncio.create_task(run_agent())
    speech_task = asyncio.create_task(speech())
    try:
        while (msg := await out.get()) is not None:
            yield _line(msg)
    finally:
        speech_task.cancel()


def _stream(lines) -> StreamingResponse:
    return StreamingResponse(lines, media_type="application/x-ndjson")


async def _just(*objs: dict):
    for o in objs:
        yield _line(o)


@app.post("/api/ask")
async def ask(body: Say):
    text, lang = body.text, reply_lang(body.lang or VOICE["lang"])
    if body.wake:
        heard, question = split_wake(text)
        if not heard:
            print(f"[voice] heard {text!r}: ignored, no wake word")
            return _stream(_just({"type": "ignored", "transcript": text}))
        if not question:                  # just "assistant": the browser chimes and waits
            print(f"[voice] heard {text!r}: wake word, waiting for the question")
            return _stream(_just({"type": "wake", "transcript": text}))
        text = question
        print(f"[voice] heard {text!r}: asking the agent")
    return _stream(_answer(text, body.speak, lang))


# "assistant", "hey assistant", "Cartesia AI assistant", "oye asistente"... at the start of an
# utterance. The leading "a" is optional because the browser can clip the first syllable.
WAKE = re.compile(r"^\W*(?:(?:hey|hi|ok|okay|oye|hola)\W+)?(?:cartesia\W+)?(?:a\.?\s?i\.?\W+)?"
                  r"(?:a?ss?ist[ae]n(?:t|ts|ce)|a\s+sis\s?tant|assist\s+and|a?sis?tente)\b[^\w¿¡]*", re.I)   # keeps Spanish ¿ ¡


def split_wake(transcript: str) -> tuple[bool, str]:
    """(wake word heard, the question after it)."""
    m = WAKE.match(transcript)
    return (True, transcript[m.end():].strip()) if m else (False, "")


@app.post("/api/voice")
async def voice_in(audio: UploadFile, lang: str = Form("")):
    """Push-to-talk: one recorded clip -> Ink-Whisper -> the agent, streamed back.
    Hands-free listening doesn't come here: the browser streams to Ink 2 directly."""
    if not voice.available():
        return JSONResponse({"error": "CARTESIA_API_KEY not set"}, 400)
    lang = reply_lang(lang or VOICE["lang"])
    try:
        transcript = await voice.transcribe(await audio.read(), audio.filename or "speech.webm", lang)
    except Exception as e:
        print(f"[voice] speech-to-text failed: {e!r}")
        return JSONResponse({"error": f"speech-to-text failed: {e}"}, 502)
    if not transcript:
        print("[voice] heard nothing")
        return _stream(_just({"type": "transcript", "text": ""}, {"type": "done", "reply": ""}))
    print(f"[voice] heard {transcript!r}: asking the agent")

    async def lines():
        yield _line({"type": "transcript", "text": transcript})
        async for line in _answer(transcript, True, lang):
            yield line
    return _stream(lines())


@app.post("/api/stt-token")
async def stt_token():
    """Short-lived token for the browser's Ink 2 socket (the API key stays here)."""
    if not voice.available():
        return JSONResponse({"error": "CARTESIA_API_KEY not set"}, 400)
    try:
        token = await voice.access_token()
    except Exception as e:
        print(f"[voice] access token failed: {e!r}")
        return JSONResponse({"error": f"access token failed: {e}"}, 502)
    return {"token": token, "version": voice.VERSION, "keyterms": voice.KEYTERMS}


class VoiceSettings(BaseModel):
    lang: str


@app.post("/api/voice-settings")
def voice_settings(body: VoiceSettings):
    if body.lang not in ("en", "es", "both"):
        return JSONResponse({"error": "lang must be en, es or both"}, 400)
    VOICE["lang"] = body.lang
    return VOICE


# ---- spoken alarms ---------------------------------------------------------------
def _to_clients(msg: dict) -> None:
    for q in list(clients):
        q.put_nowait(msg)


async def announce_alarm(kind: str, say: dict | None = None) -> None:
    """Speak an alarm on every open dashboard, straight from the device event: no LLM
    in the path, urgent voice, and in Spanish too if the dashboard asks for it."""
    lines = voice.alarm_lines(kind, VOICE["lang"], say)
    for i, (text, lang) in enumerate(lines):
        clip = uuid.uuid4().hex
        try:
            if not voice.available():
                raise RuntimeError("no CARTESIA_API_KEY")
            first = True
            async for pcm in voice.stream_speech(text, lang, style="alarm"):
                _to_clients(_audio(pcm, alarm=True, clip=clip, first=first))
                first = False
        except Exception as e:
            print(f"[alarm] streaming speech failed ({e!r}); browser voice instead")
            _to_clients({"type": "alarm_say", "lines": lines[i:]})
            return


async def alarm_speaker() -> None:
    # Alarms are spoken one after another by a worker, so the state queue keeps
    # draining while one is being said (runtime drops messages for full queues).
    q: asyncio.Queue = asyncio.Queue()
    kinds: asyncio.Queue = asyncio.Queue()

    async def worker():
        while True:
            kind, say = await kinds.get()
            try:
                await announce_alarm(kind, say)
            except Exception as err:
                print("[alarm]", repr(err))

    rt.subscribers.add(q)
    task = asyncio.create_task(worker())
    try:
        while True:
            msg = await q.get()
            for e in msg.get("events", []):
                say = (e.get("data") or {}).get("say")
                if e["severity"] == "alarm" or say:     # warnings with their own words too
                    kinds.put_nowait((e["kind"], say))
    finally:
        task.cancel()
        rt.subscribers.discard(q)


class OrderIn(BaseModel):
    rate_ml_h: float
    drop_factor: int
    vtbi_ml: float
    bag_volume_ml: float
    drug: str = "fluid"


@app.get("/api/history")
def history(minutes: int = 30):
    """Flow history, no-drop gaps and time to empty from the Tiger Data aggregates
    (read-only agent role). 503 when Timescale isn't configured."""
    from . import timeseries
    q = timeseries.queries()
    if not q.available:
        return JSONResponse({"error": "no AGENT_DATABASE_URL: history needs Tiger Data"}, 503)
    try:
        return {"flow": q.flow_history(minutes), "gaps": q.flow_gaps(min(minutes, 60)),
                "time_to_empty": q.time_to_empty()}
    except LookupError as e:
        return JSONResponse({"error": str(e)}, 404)


@app.post("/api/order")
def order(body: OrderIn):
    """Manual entry fallback, same validation as the agent path."""
    try:
        with rt.lock:
            rt.twin.start(Order(**body.model_dump()), rt.now())
    except OrderRejected as e:
        return JSONResponse({"error": str(e)}, 400)
    rt.log("order", {**body.model_dump(), "confirmation": "manual entry"})
    return {"ok": True}


@app.post("/api/{action}")
def control(action: str):
    if action not in ("pause", "resume"):
        return JSONResponse({"error": "unknown"}, 404)
    with rt.lock:
        getattr(rt.twin, action)(rt.now())
    return {"mode": rt.twin.mode.value}


@app.post("/api/sim/{action}")
def sim(action: str):
    return {"result": rt.device.disturb(action)}


def main():
    import uvicorn
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="simulated plant + synthetic camera")
    ap.add_argument("--port", help="Pico serial port, e.g. /dev/tty.usbmodem101")
    ap.add_argument("--monitor", action="store_true",
                    help="webcam drop counting only: no Pico, no valve")
    ap.add_argument("--camera", default="0", help="webcam index or video file; 'none' to disable")
    ap.add_argument("--head", type=float, help="bag height above the site in cm")
    ap.add_argument("--http", type=int, default=8000)
    a = ap.parse_args()
    if sum(map(bool, (a.sim, a.port, a.monitor))) != 1:
        ap.error("pass exactly one of --sim, --port or --monitor")
    ARGS.sim, ARGS.port, ARGS.head, ARGS.monitor = a.sim, a.port, a.head, a.monitor
    ARGS.camera = None if a.camera == "none" else int(a.camera) if a.camera.isdigit() else a.camera
    if a.monitor and ARGS.camera is None:
        ap.error("--monitor needs a camera")
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", a.http)) == 0:
            raise SystemExit(
                f"Port {a.http} is already in use: another DripTwin server is probably still running "
                f"(often one paused with Ctrl+Z). Stop it with:  pkill -9 -f driptwin.server\n"
                f"or start this one on another port:  --http {a.http + 1}")
    uvicorn.run(app, host="0.0.0.0", port=a.http)


if __name__ == "__main__":
    main()
