"""Voice pipeline without the real Cartesia or Claude: spoken-text rewriting, the
order confirmation gate, Sonic streaming against a fake socket, and the server's
streamed replies and spoken alarms."""

import asyncio
import base64
import json

import pytest

from driptwin import voice
from driptwin.agent import Agent
from driptwin.twin import DripTwin


# ---- what gets said ----------------------------------------------------------------
@pytest.mark.parametrize("text, said", [
    ("Running at 120 mL/h.", "Running at 120 milliliters per hour."),
    ("40 gtt/min, 20 gtt set", "40 drops per minute, 20 drops set"),
    ("target 40 drops/min", "target 40 drops per minute"),
    ("~50 mL left, 1 mL/min", "about 50 milliliters left, 1 milliliter per minute"),
    ("Raise the bag 12 cm", "Raise the bag 12 centimeters"),
    ("1 mL", "1 milliliter"),
])
def test_normalize_english(text, said):
    assert voice.normalize(text, "en") == said


def test_normalize_spanish():
    assert voice.normalize("120 mL/h, 40 gtt/min, ~50 mL", "es") == \
        "120 mililitros por hora, 40 gotas por minuto, aproximadamente 50 mililitros"


def test_streamed_text_is_normalized_on_word_boundaries():
    async def parts():
        for p in ["Set to 120 m", "L/", "h now, 40 gtt", "/min"]:
            yield p

    async def collect():
        return "".join([t async for t in voice._speakable(parts(), "en")])
    assert asyncio.run(collect()) == "Set to 120 milliliters per hour now, 40 drops per minute"


def test_alarm_lines():
    assert voice.alarm_lines("occlusion", "en")[0][0].startswith("Alarm. Line blocked")
    both = voice.alarm_lines("occlusion", "both")
    assert [lang for _, lang in both] == ["en", "es"] and both[1][0].startswith("Alarma")
    assert voice.alarm_lines("something_new", "es") == [(voice.ALARM_FALLBACK["es"], "es")]


def test_alarm_style_drops_emotion_outside_english():
    voice._dict_id = "dict_1"
    try:
        en, es = voice._request("en", "calm"), voice._request("es", "calm")
    finally:
        voice._dict_id = None
    assert en["generation_config"]["emotion"] == "calm" and en["pronunciation_dict_id"] == "dict_1"
    assert "emotion" not in es["generation_config"] and "pronunciation_dict_id" not in es


# ---- order read-back gate ------------------------------------------------------------
class OfflineAgent(Agent):
    def _connect(self):
        return None


ORDER = {"rate_ml_h": 120, "drop_factor": 20, "vtbi_ml": 500, "bag_volume_ml": 1000,
         "drug": "normal saline"}


def test_start_needs_confirmation_in_a_later_turn():
    a = OfflineAgent(DripTwin(head_cm=90))
    a._begin("Set 120 mL per hour, 20-drop set, 500 mL of saline", "en")
    out, err = a._tool("propose_order", ORDER)
    assert not err and json.loads(out)["drops_per_min"] == 31.5     # measured 20-drop set
    out, err = a._tool("start_infusion", {"confirmation_quote": "yes"})
    assert err and "next message" in out                   # same turn as the read-back

    a._begin("Hmm, what was the rate again?", "en")
    out, err = a._tool("start_infusion", {"confirmation_quote": "confirm"})
    assert err and "hasn't confirmed" in out               # model's quote isn't enough
    assert a.twin.order is None

    a._begin("Confirm.", "en")
    out, err = a._tool("start_infusion", {"confirmation_quote": "Confirm."})
    assert not err and a.twin.order is not None


def test_spanish_confirmation_and_reply_language():
    a = OfflineAgent(DripTwin(head_cm=90))
    a._begin("Ciento veinte mililitros por hora", "es")
    a._tool("propose_order", ORDER)
    ctx = a._begin("Sí, confirmo", "es")
    assert ctx.startswith("[reply in Spanish]")
    assert not a._tool("start_infusion", {"confirmation_quote": "Sí, confirmo"})[1]


# ---- Sonic streaming against a fake Cartesia socket -------------------------------------
def test_stream_speech_protocol(monkeypatch):
    websockets = pytest.importorskip("websockets")
    received = []

    async def fake_cartesia(ws):
        async for raw in ws:
            msg = json.loads(raw)
            received.append(msg)
            if msg.get("transcript"):
                await ws.send(json.dumps({"type": "chunk", "context_id": msg["context_id"],
                                          "data": base64.b64encode(b"\x01\x00" * 4).decode(),
                                          "done": False}))
            if msg.get("continue") is False:
                await ws.send(json.dumps({"type": "done", "context_id": msg["context_id"],
                                          "done": True}))

    async def scenario():
        async with websockets.serve(fake_cartesia, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            monkeypatch.setattr(voice, "WS_API", f"ws://127.0.0.1:{port}")
            monkeypatch.setenv("CARTESIA_API_KEY", "test")
            monkeypatch.setattr(voice, "_sock", voice._Socket())

            async def deltas():
                for d in ["Line ", "blocked at 40 gtt", "/min."]:
                    yield d
            audio = b"".join([c async for c in voice.stream_speech(deltas(), "en", "alarm")])
            again = b"".join([c async for c in voice.stream_speech("Second.", "es")])
            await voice._sock.ws.close()
            return audio, again

    audio, again = asyncio.run(scenario())
    first = [m for m in received if m["context_id"] == received[0]["context_id"]]
    texts = [m for m in first if m["transcript"]]
    assert audio == b"\x01\x00" * 4 * len(texts) and again == b"\x01\x00" * 4
    assert "".join(m["transcript"] for m in first) == "Line blocked at 40 drops per minute."
    assert [m["continue"] for m in first] == [True] * len(texts) + [False]
    assert first[0]["generation_config"]["speed"] > 1 and first[0]["language"] == "en"
    assert first[0]["output_format"] == {"container": "raw", "encoding": "pcm_s16le",
                                         "sample_rate": voice.SAMPLE_RATE}
    assert received[-1]["language"] == "es"          # second reply reused the same socket


# ---- server: streamed replies and spoken alarms ----------------------------------------
@pytest.fixture
def app(monkeypatch):
    from fastapi.testclient import TestClient
    from driptwin import server
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(server.ARGS, "camera", None)
    spoken = []

    async def fake_stream(parts, lang="en", style="calm"):
        text = parts if isinstance(parts, str) else "".join([p async for p in parts])
        spoken.append((text, lang, style))
        yield b"\x00\x01" * 10

    async def no_warm_up():
        pass
    monkeypatch.setattr(voice, "stream_speech", fake_stream)
    monkeypatch.setattr(voice, "warm_up", no_warm_up)
    with TestClient(server.app) as client:
        yield client, server, spoken


def lines(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_ask_streams_text_then_audio_then_done(app):
    client, server, spoken = app
    out = lines(client.post("/api/ask", json={"text": "how is it going?"}))
    kinds = [m["type"] for m in out]
    assert kinds[0] == "text" and "audio" in kinds and kinds[-1] == "done"
    assert out[-1]["spoken"] and out[-1]["reply"] == "No infusion running."
    assert spoken[-1] == ("No infusion running.", "en", "calm")


def test_ask_wake_word_filter(app):
    client, _, _ = app
    assert lines(client.post("/api/ask", json={"text": "nice weather", "wake": True}))[0]["type"] == "ignored"
    assert lines(client.post("/api/ask", json={"text": "Assistant.", "wake": True}))[0]["type"] == "wake"
    out = lines(client.post("/api/ask", json={"text": "Assistant, status?", "wake": True}))
    assert out[-1]["type"] == "done"


def test_voice_language_setting_drives_replies(app):
    client, _, spoken = app
    assert client.post("/api/voice-settings", json={"lang": "fr"}).status_code == 400
    client.post("/api/voice-settings", json={"lang": "es"})
    client.post("/api/ask", json={"text": "estado"})
    assert spoken[-1][1] == "es"


def test_alarm_is_spoken_to_every_dashboard(app):
    client, server, spoken = app
    client.post("/api/voice-settings", json={"lang": "both"})
    q = asyncio.Queue()
    server.clients.add(q)
    try:
        client.portal.call(server.announce_alarm, "occlusion")
    finally:
        server.clients.discard(q)
    msgs = [q.get_nowait() for _ in range(q.qsize())]
    assert all(m["type"] == "audio" and m["alarm"] for m in msgs) and len(msgs) == 2
    assert [m["first"] for m in msgs] == [True, True]       # one clip per language
    assert [(lang, style) for _, lang, style in spoken[-2:]] == [("en", "alarm"), ("es", "alarm")]


def test_event_words_replace_the_fixed_alarm_line():
    say = {"en": "Warning. Flow is high.", "es": "Atención. Flujo alto."}
    assert voice.alarm_lines("high_flow", "both", say) == [("Warning. Flow is high.", "en"),
                                                           ("Atención. Flujo alto.", "es")]
    assert voice.alarm_lines("high_flow", "en") == [(voice.ALARM_FALLBACK["en"], "en")]


def test_warning_with_words_is_spoken_by_the_server(app):
    client, server, spoken = app
    import time as _t
    client.post("/api/voice-settings", json={"lang": "en"})     # earlier tests may leave "both"
    server.rt._broadcast({"type": "state", "events": [
        {"kind": "no_flow", "severity": "warn", "message": "x",
         "data": {"say": {"en": "Nothing is dripping. The valve is closed."}}},
        {"kind": "probe", "severity": "warn", "message": "quiet", "data": {}}]})
    for _ in range(50):                          # the speaker worker runs on the app's loop
        if spoken:
            break
        _t.sleep(0.05)
    assert [s[:3] for s in spoken] == [("Nothing is dripping. The valve is closed.", "en", "alarm")]
