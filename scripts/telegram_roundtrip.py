#!/usr/bin/env python
"""Live Telegram phone round-trip acceptance check (SWARM_V2 §4.6 — manual leg).

Runs the REAL production wiring: a SwarmRunner whose in-process TelegramBridge talks to the
real Telegram Bot API. You run this in YOUR shell (where the bot token is already exported for
hermes), watch your phone, and send a message — proving the same-session bidirectional mirror
end to end. The secret never enters the assistant chat: it is read from the environment here.

Usage (in your own terminal, from the repo root):

    SWARM_TELEGRAM_TOKEN="$TELEGRAM_BOT_TOKEN" \
    SWARM_TELEGRAM_CHAT_IDS="$TELEGRAM_CHAT_ID" \
    PYTHONPATH=. ~/.hermes/hermes-agent/venv/bin/python scripts/telegram_roundtrip.py [seconds]

Then on your phone:
  * you should receive a "bridge is live" message (OUTBOUND mirror working),
  * reply with  /tasks   → you get the queue snapshot back (INBOUND routing working),
  * or send a goal like  summarize what swarm-agent does  → it runs and the result arrives.
Ctrl-C to stop early. Paste the terminal output back to confirm the round-trip.
"""
from __future__ import annotations

import os
import sys
import time


def _resolve(name, *fallbacks):
    for key in (name, *fallbacks):
        v = os.environ.get(key)
        if v:
            return v
    return None


def _preflight(token: str) -> bool:
    """Confirm the token works and warn about the #1 failure mode: a webhook (or another
    getUpdates consumer such as a running hermes Telegram gateway) on the same bot, which
    silently blocks our long-poll. Returns False only on a clearly-broken token."""
    import json
    import urllib.request

    def call(method):
        try:
            with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}",
                                        timeout=10) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    me = call("getMe")
    if not me.get("ok"):
        print(f"PREFLIGHT FAIL: getMe rejected the token ({me.get('error') or me}).",
              file=sys.stderr)
        return False
    uname = (me.get("result") or {}).get("username")
    print(f"preflight: bot @{uname} reachable")
    wh = call("getWebhookInfo")
    url = ((wh.get("result") or {}).get("url") or "") if wh.get("ok") else ""
    if url:
        print(f"⚠️  WARNING: a webhook is set ({url}). getUpdates (our inbound) will be "
              f"blocked until it is removed (deleteWebhook) or use a different bot token.",
              file=sys.stderr)
    else:
        print("preflight: no webhook set (getUpdates inbound path is clear). If hermes' own "
              "Telegram gateway is polling this same bot, stop it first to avoid a 409 conflict.")
    return True


def main() -> int:
    token = _resolve("SWARM_TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN")
    chat = _resolve("SWARM_TELEGRAM_CHAT_IDS", "TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("ERROR: set SWARM_TELEGRAM_TOKEN and SWARM_TELEGRAM_CHAT_IDS (or rely on "
              "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID being exported in this shell).",
              file=sys.stderr)
        print("Nothing is sent without an allowlisted chat id.", file=sys.stderr)
        return 2
    # Configure the bridge via env BEFORE constructing the runner, so runner._telegram (the
    # real production path) picks it up and start_manager() brings it up alongside the manager.
    os.environ["SWARM_TELEGRAM_TOKEN"] = token
    os.environ["SWARM_TELEGRAM_CHAT_IDS"] = chat

    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120

    if not _preflight(token):
        return 2

    from swarm_agent.runner import SwarmRunner
    runner = SwarmRunner(warm=False, admission="static", gate_start=16)
    bridge = runner._telegram
    if bridge is None or not bridge.configured:
        print("ERROR: Telegram bridge did not configure (token/allowlist).", file=sys.stderr)
        return 2
    runner.setup()
    runner.start_manager()                      # starts the completion manager AND the bridge
    print(f"bridge configured={bridge.configured} allowlist={sorted(bridge.allowed, key=str)} "
          f"primary_chat={bridge.primary_chat}")

    # OUTBOUND proof: emit a status event; the bridge tails the logbook and mirrors it to phone.
    runner.emit("status", text="🟢 swarm-agent Telegram bridge is live — reply with /tasks, "
                "or send a goal. (live round-trip test)")

    print(f"Listening for {duration}s. Send a message from your phone now. Ctrl-C to stop.\n")
    seen = 0
    deadline = time.time() + duration
    try:
        while time.time() < deadline:
            try:
                ev = runner.events.get(timeout=1.0)
            except Exception:
                continue
            seen += 1
            kind = ev.get("kind")
            txt = (ev.get("text") or ev.get("goal") or ev.get("error") or "")[:120]
            print(f"  [{kind}] {txt}".rstrip())
    except KeyboardInterrupt:
        print("\n(stopped)")
    finally:
        runner.shutdown()
    print(f"\nDone. {seen} session events observed. "
          f"If your phone received the greeting AND your reply was acted on, §4.6 round-trip PASSES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
