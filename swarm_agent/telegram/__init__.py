"""In-process Telegram session mirror for swarm-agent (SWARM_V2 Phase 4).

Self-contained: nothing here imports HermesAgent or python-telegram-bot at runtime (the Bot API
transport uses the standard library only — §6.2)."""
from .bridge import TelegramBridge
from .inbound import route_inbound
from .render import LogTailer, Renderer

__all__ = ["TelegramBridge", "route_inbound", "Renderer", "LogTailer"]
