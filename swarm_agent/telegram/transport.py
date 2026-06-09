"""Self-contained Telegram Bot API transport (no external deps, no HermesAgent import).

Deliberately vendored/self-contained per SWARM_V2 §6.2: this talks to the Bot API over the
standard library (``urllib``) only — it does NOT import python-telegram-bot, HermesAgent's
``gateway/platforms/telegram*``, or anything under ``~/.hermes``. The bridge stays a drop-in
part of swarm-agent so ``git pull`` in the HermesAgent repo can never affect it.

Interface (what TelegramBridge depends on; a fake implements the same in tests):
  send_message(chat_id, text) -> message_id | None
  edit_message(chat_id, message_id, text) -> bool
  get_updates(offset, timeout) -> list[{"update_id", "chat_id", "text"}]
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional


class HttpTelegramTransport:
    def __init__(self, token: str, *, timeout: float = 30.0) -> None:
        self._base = f"https://api.telegram.org/bot{token}"
        self._timeout = timeout

    def _call(self, method: str, params: dict) -> Optional[dict]:
        url = f"{self._base}/{method}"
        data = urllib.parse.urlencode(params).encode()
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None   # network/Bot-API failure → soft-degrade (bridge keeps running)

    def send_message(self, chat_id, text: str) -> Optional[int]:
        res = self._call("sendMessage", {"chat_id": chat_id, "text": text})
        if res and res.get("ok"):
            return res["result"]["message_id"]
        return None

    def edit_message(self, chat_id, message_id, text: str) -> bool:
        res = self._call("editMessageText",
                         {"chat_id": chat_id, "message_id": message_id, "text": text})
        return bool(res and res.get("ok"))

    def get_updates(self, offset: int, timeout: float = 25.0) -> list[dict]:
        res = self._call("getUpdates", {"offset": offset, "timeout": int(timeout)})
        out: list[dict] = []
        if res and res.get("ok"):
            for u in res["result"]:
                msg = u.get("message") or u.get("edited_message") or {}
                out.append({"update_id": u.get("update_id"),
                            "chat_id": (msg.get("chat") or {}).get("id"),
                            "text": msg.get("text") or ""})
        return out
