"""Pre-demo check of the voice agent with the real API keys. Never prints a key.

    python -m driptwin.selfcheck

Checks: which keys .env provides, Cartesia text-to-speech and speech-to-text,
and one real question to the agent (Claude, or Gemini if that's the only key)
against a simulated drip running at 40 drops per minute.
"""

import asyncio
import os
import time

from . import voice
from .agent import Agent, GeminiAgent
from .server import ENV_FILE, load_env
from .twin import DripTwin, Order

KEYS = ("ANTHROPIC_API_KEY", "CARTESIA_API_KEY", "GEMINI_API_KEY")


def check_voice() -> bool:
    async def run() -> None:
        t = time.perf_counter()
        mp3 = await voice.speak("Running at forty drops per minute.")
        print(f"  text-to-speech   OK  {len(mp3) // 1024} KB in {time.perf_counter() - t:.1f} s")
        t = time.perf_counter()
        text = await voice.transcribe(mp3, "check.mp3")
        print(f"  speech-to-text   OK  heard {text!r} in {time.perf_counter() - t:.1f} s")
    try:
        asyncio.run(run())
        return True
    except Exception as e:
        print(f"  Cartesia FAILED: {e!r}"[:300])
        return False


def check_agent() -> bool:
    twin = DripTwin(head_cm=80.0)
    twin.start(Order(120, 20, 1000, 1000, "saline"), 0.0)
    for i in range(1, 40):
        twin.on_drop(i * 1.5)
    twin.tick(60.0)
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    agent = (Agent if use_claude else GeminiAgent)(twin, clock=lambda: 60.0)
    t = time.perf_counter()
    try:
        reply = agent.ask("How is the drip doing?")
    except Exception as e:
        print(f"  {agent.NAME} FAILED: {e!r}"[:300])
        return False
    print(f"  {agent.NAME:<16} OK  {time.perf_counter() - t:.1f} s: {reply!r}")
    return True


def main() -> None:
    load_env()
    print(f"keys ({ENV_FILE}):")
    for k in KEYS:
        print(f"  {k:<18} {'set' if os.environ.get(k) else 'MISSING'}")
    print("Cartesia:")
    voice_ok = voice.available() and check_voice()
    if not voice.available():
        print("  skipped: no CARTESIA_API_KEY")
    print("agent:")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  no ANTHROPIC_API_KEY: testing the Gemini fallback instead of Claude")
    agent_ok = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    agent_ok = agent_ok and check_agent()
    print("ALL GOOD" if voice_ok and agent_ok else "SOMETHING FAILED OR SKIPPED (see above)")


if __name__ == "__main__":
    main()
