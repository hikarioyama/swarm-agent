"""LanceDB-backed conversation recall — CPU hybrid (vector + BM25/FTS) over past turns.

Path B (non-invasive): the front door (chat/router) and planner no longer FORGET beyond
the recent N-turn snippet. Every turn is indexed into a persistent LanceDB store (survives
restart, spans sessions). On each chat/plan turn we hybrid-search the user's message and
inject the most relevant OLDER turns into the prompt — recent stays verbatim, older is
referenced on demand. This is the "随時参照" the user asked for, without touching HermesAgent.

Embeddings: fastembed multilingual MiniLM (384-d). **CPU-ONLY by design** (GPU is reserved
for the Step-3.7 inference server; onnxruntime CPUExecutionProvider, low thread count).

Everything here is FAIL-SOFT: if a dependency / model / store is unavailable or any call
errors, recall degrades to "" / no-op and NEVER breaks the front door. Tunable via env:
  SWARM_RECALL=0            disable entirely
  SWARM_RECALL_PATH=...     LanceDB dir (default ~/.cache/swarm-agent/recall.lance)
  SWARM_RECALL_MODEL=...    fastembed model (default multilingual MiniLM, 384-d)
  SWARM_RECALL_TOPK=4       max injected turns
  SWARM_RECALL_THREADS=2    onnxruntime intra-op threads (keep low; CPU is shared)
"""
from __future__ import annotations

import os
import re
import glob
import json
import datetime as _dt
import threading
import time
from typing import Iterable, Optional

_ENABLED = os.environ.get("SWARM_RECALL", "1") not in ("0", "false", "False")
_PATH = os.environ.get("SWARM_RECALL_PATH",
                       os.path.expanduser("~/.cache/swarm-agent/recall.lance"))
_MODEL = os.environ.get("SWARM_RECALL_MODEL",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_DIM = int(os.environ.get("SWARM_RECALL_DIM", "384"))
_TOPK = int(os.environ.get("SWARM_RECALL_TOPK", "4"))
_THREADS = int(os.environ.get("SWARM_RECALL_THREADS", "2"))
_MAX_CHARS = int(os.environ.get("SWARM_RECALL_MAX_CHARS", "1500"))  # per-turn cap in the block
_RRF_K = 60  # reciprocal-rank-fusion constant


def _now() -> float:
    return time.time()


# ── Japan Standard Time (UTC+9), fixed offset (no DST) ───────────────────────────
_JST = _dt.timezone(_dt.timedelta(hours=9))


def _jst(ts: Optional[float] = None) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ts if ts is not None else time.time(), _JST)


def _jst_date_str(ts: Optional[float] = None) -> str:
    """JST calendar date, e.g. '2026-06-05'."""
    return _jst(ts).strftime("%Y-%m-%d")


def _day_bounds(d: _dt.date) -> tuple[float, float]:
    """[start, end) epoch seconds for a JST calendar day."""
    start = _dt.datetime(d.year, d.month, d.day, tzinfo=_JST)
    return start.timestamp(), (start + _dt.timedelta(days=1)).timestamp()


def parse_jst_window(query: str, now: Optional[_dt.datetime] = None):
    """Map a temporal phrase in the query to a JST [start, end) epoch window, or None.

    Handles common Japanese/English relative terms (今日/昨日/一昨日/今週/先週/先々週/
    今月/先月, today/yesterday/this|last week|month), 'N日前'/'N days ago', and explicit
    dates (YYYY-MM-DD, M月D日). Returns None when no temporal cue is present (so normal
    queries are unaffected). Week starts Monday (ISO)."""
    if not query:
        return None
    q = query.lower()
    now = now or _jst()
    today = now.date()

    def span(d0: _dt.date, d1: _dt.date):  # inclusive d0..d1
        return _day_bounds(d0)[0], _day_bounds(d1)[1]

    # explicit YYYY-MM-DD
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", q)
    if m:
        try:
            d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return span(d, d)
        except ValueError:
            pass
    # explicit Japanese M月D日 (assume current year)
    m = re.search(r"(\d{1,2})月(\d{1,2})日", query)
    if m:
        try:
            d = _dt.date(today.year, int(m.group(1)), int(m.group(2)))
            return span(d, d)
        except ValueError:
            pass
    # N日前 / N days ago
    m = re.search(r"(\d+)\s*日前", query) or re.search(r"(\d+)\s*days?\s*ago", q)
    if m:
        d = today - _dt.timedelta(days=int(m.group(1)))
        return span(d, d)
    if any(w in query for w in ("一昨日", "おととい")) or "day before yesterday" in q:
        d = today - _dt.timedelta(days=2)
        return span(d, d)
    if "昨日" in query or "きのう" in query or "yesterday" in q:
        d = today - _dt.timedelta(days=1)
        return span(d, d)
    if "今日" in query or "きょう" in query or "today" in q:
        return span(today, today)
    monday = today - _dt.timedelta(days=today.weekday())
    if "先々週" in query:
        s = monday - _dt.timedelta(days=14)
        return span(s, s + _dt.timedelta(days=6))
    if "先週" in query or "last week" in q:
        s = monday - _dt.timedelta(days=7)
        return span(s, s + _dt.timedelta(days=6))
    if "今週" in query or "this week" in q:
        return span(monday, monday + _dt.timedelta(days=6))
    first = today.replace(day=1)
    if "先月" in query or "last month" in q:
        last_prev = first - _dt.timedelta(days=1)
        return span(last_prev.replace(day=1), last_prev)
    if "今月" in query or "this month" in q:
        nxt = (first + _dt.timedelta(days=32)).replace(day=1)
        return span(first, nxt - _dt.timedelta(days=1))
    return None


def _clip(text: str, n: int = _MAX_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


class RecallStore:
    """Persistent conversation index with CPU hybrid search. Thread-safe and fail-soft."""

    def __init__(self, path: str = _PATH, model_name: str = _MODEL,
                 enabled: bool = _ENABLED) -> None:
        self._path = path
        self._model_name = model_name
        self._enabled = bool(enabled)
        self._db = None
        self._tbl = None
        self._emb = None
        self._fts_ready = False
        self._ready = False
        self._lock = threading.Lock()
        self._add_lock = threading.Lock()

    # ── lazy init (model load + table open); never raises ────────────────────────
    def _ensure(self) -> bool:
        if self._ready:
            return True
        if not self._enabled:
            return False
        with self._lock:
            if self._ready:
                return True
            try:
                import pyarrow as pa  # noqa
                import lancedb
                from fastembed import TextEmbedding
                # CPU-ONLY: do not let onnxruntime grab a GPU; keep threads modest so the
                # embedder never competes with the swarm's worker threads for the box.
                try:
                    self._emb = TextEmbedding(model_name=self._model_name,
                                              providers=["CPUExecutionProvider"],
                                              threads=_THREADS)
                except TypeError:
                    # older/newer fastembed signature — fall back to defaults (still CPU,
                    # since only onnxruntime CPU wheel is installed)
                    self._emb = TextEmbedding(model_name=self._model_name)
                os.makedirs(self._path, exist_ok=True)
                self._db = lancedb.connect(self._path)
                # 0.33 table_names() returns a flat list[str] (deprecation-warned but
                # correct). list_tables() returns a NESTED/paginated shape that is not
                # set()-able, so prefer table_names(); normalize defensively either way.
                try:
                    raw = self._db.table_names()
                except AttributeError:
                    raw = self._db.list_tables()
                names = set()
                for t in (raw or []):
                    if isinstance(t, str):
                        names.add(t)
                    elif isinstance(t, (list, tuple)):
                        names.update(x for x in t if isinstance(x, str))
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("session_id", pa.string()),
                    pa.field("turn_idx", pa.int64()),
                    pa.field("role", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("ts", pa.float64()),          # epoch seconds (UTC)
                    pa.field("date_jst", pa.string()),     # JST calendar date 'YYYY-MM-DD'
                    pa.field("vector", pa.list_(pa.float32(), _DIM)),
                ])
                if "turns" in names:
                    self._tbl = self._db.open_table("turns")
                    # schema evolution: an older store may predate date_jst → recreate
                    if "date_jst" not in set(self._tbl.schema.names):
                        self._db.drop_table("turns")
                        self._tbl = self._db.create_table("turns", schema=schema)
                else:
                    self._tbl = self._db.create_table("turns", schema=schema)
                self._ready = True
            except Exception as e:  # pragma: no cover - environment dependent
                self._enabled = False
                self._log(f"recall disabled (init failed): {type(e).__name__}: {e}")
            return self._ready

    def _log(self, msg: str) -> None:
        if os.environ.get("SWARM_RECALL_DEBUG"):
            print(f"[recall] {msg}", flush=True)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._emb.embed(texts)]

    def _ensure_fts(self) -> None:
        """(Re)build the native FTS index. Cheap for a conversation-sized table; called
        lazily so it always covers the latest rows. Best-effort."""
        # ngram base tokenizer: the DEFAULT (simple/whitespace) tokenizer cannot segment
        # Japanese (no spaces), so 'ビルドID' matched ZERO rows. Character n-grams (2..3)
        # make CJK substrings matchable, which is what lets BM25 rescue the semantic search
        # when the embedder is diluted by a verbose conversational query.
        try:
            self._tbl.create_fts_index(
                "text", replace=True, use_tantivy=False,
                base_tokenizer="ngram", ngram_min_length=2, ngram_max_length=3,
                prefix_only=False, lower_case=True, stem=False, remove_stop_words=False)
            self._fts_ready = True
        except Exception as e:
            # fall back to a plain native index, then tantivy, so FTS still works for ASCII
            try:
                self._tbl.create_fts_index("text", replace=True, use_tantivy=False)
                self._fts_ready = True
            except Exception:
                try:
                    self._tbl.create_fts_index("text", replace=True)
                    self._fts_ready = True
                except Exception as e2:
                    self._log(f"fts index build failed: {e} / {e2}")

    # ── write ────────────────────────────────────────────────────────────────────
    def add(self, role: str, text: str, session_id: str, turn_idx: int,
            ts: Optional[float] = None) -> None:
        text = (text or "").strip()
        if not text or not self._ensure():
            return
        try:
            with self._add_lock:
                vec = self._embed([text])[0]
                ts_v = float(ts if ts is not None else _now())
                row = {
                    "id": f"{session_id}:{turn_idx}",
                    "session_id": session_id,
                    "turn_idx": int(turn_idx),
                    "role": role,
                    "text": text,
                    "ts": ts_v,
                    "date_jst": _jst_date_str(ts_v),
                    "vector": vec,
                }
                try:
                    (self._tbl.merge_insert("id")
                         .when_matched_update_all()
                         .when_not_matched_insert_all()
                         .execute([row]))
                except Exception:
                    self._tbl.add([row])  # fallback (may duplicate; ids still distinct)
                self._fts_ready = False  # new row → FTS stale; rebuilt on next search
        except Exception as e:
            self._log(f"add failed: {type(e).__name__}: {e}")

    def add_async(self, role: str, text: str, session_id: str, turn_idx: int,
                  ts: Optional[float] = None) -> None:
        """Fire-and-forget indexing so a turn is never blocked on embedding."""
        if not self._enabled or not (text or "").strip():
            return
        threading.Thread(target=self.add, args=(role, text, session_id, turn_idx, ts),
                         daemon=True).start()

    # ── read (hybrid: vector + FTS, fused by RRF; optional JST time window) ───────
    def search(self, query: str, k: int = _TOPK,
               exclude_texts: Iterable[str] = ()) -> list[dict]:
        query = (query or "").strip()
        if not query or not self._ensure():
            return []
        try:
            n = max(self._tbl.count_rows(), 0)
        except Exception:
            n = 1
        if n == 0:
            return []
        try:
            # Temporal cue ("先週"/"今日"/"6月3日"/...) → restrict to that JST day-range so
            # "先週言ったこと" hits. No cue → window=None and the time filter is skipped.
            window = parse_jst_window(query)
            where = None
            if window:
                where = f"ts >= {window[0]} AND ts < {window[1]}"
            # widen the candidate pool when we're going to time-filter
            lim = k * (8 if where else 4)
            qv = self._embed([query])[0]

            def _run(builder):
                try:
                    if where:
                        try:
                            builder = builder.where(where, prefilter=True)
                        except TypeError:
                            builder = builder.where(where)
                    return builder.limit(lim).to_list()
                except Exception as e:
                    self._log(f"search leg failed: {e}")
                    return []

            # vector half (exact/flat for small tables — no index needed)
            vres = _run(self._tbl.search(qv))
            # keyword half (BM25/FTS, ngram-tokenized) — rebuild index if stale
            if not self._fts_ready:
                self._ensure_fts()
            fres = _run(self._tbl.search(query, query_type="fts"))
            # if a time window matched nothing at all, fall back to no-window (don't go blank)
            if where and not vres and not fres:
                where = None
                vres = self._tbl.search(qv).limit(k * 4).to_list()
                try:
                    fres = self._tbl.search(query, query_type="fts").limit(k * 4).to_list()
                except Exception:
                    fres = []
            fused = self._rrf(vres, fres)
            excl = {(_clip(t)).strip() for t in exclude_texts}
            out, seen = [], set()
            for r in fused:
                rid = r.get("id")
                txt = _clip(r.get("text", ""))
                if rid in seen or txt in excl or not txt:
                    continue
                seen.add(rid)
                out.append(r)
                if len(out) >= k:
                    break
            return out
        except Exception as e:
            self._log(f"search failed: {type(e).__name__}: {e}")
            return []

    @staticmethod
    def _rrf(vres: list[dict], fres: list[dict]) -> list[dict]:
        """Reciprocal-rank fusion of two ranked lists keyed by row id."""
        score: dict[str, float] = {}
        row: dict[str, dict] = {}
        for lst in (vres, fres):
            for rank, r in enumerate(lst):
                rid = r.get("id")
                if rid is None:
                    continue
                score[rid] = score.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
                row.setdefault(rid, r)
        return [row[i] for i in sorted(score, key=lambda x: -score[x])]

    def block(self, query: str, k: int = _TOPK,
              exclude_texts: Iterable[str] = ()) -> str:
        """Return a labelled, prompt-ready block of relevant earlier turns, or ""."""
        rows = self.search(query, k=k, exclude_texts=exclude_texts)
        if not rows:
            return ""
        lines = ["[earlier messages from THIS conversation, retrieved as relevant to the "
                 "user's latest message. These ARE real prior turns (dates in JST) — treat "
                 "them as authoritative history, not speculation:]"]
        for r in rows:
            d = r.get("date_jst") or ""
            stamp = f" [{d}]" if d else ""
            lines.append(f"{r.get('role','?')}{stamp}: {_clip(r.get('text',''))}")
        return "\n".join(lines)

    # ── backfill from the persisted logbook (cross-session history) ──────────────
    def backfill_from_logs(self, log_dir: Optional[str] = None,
                           max_files: int = 50) -> int:
        """Index user/assistant turns from prior logbook JSONL files. Idempotent (id =
        session:idx via merge_insert). Returns the number of turns indexed. Best-effort."""
        if not self._ensure():
            return 0
        log_dir = log_dir or os.path.expanduser("~/.cache/swarm-agent/logs")
        files = sorted(glob.glob(os.path.join(log_dir, "events-*.jsonl")))[-max_files:]
        count = 0
        for fp in files:
            per_sid: dict[str, int] = {}
            try:
                with open(fp, "r") as fh:
                    for line in fh:
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        kind = ev.get("kind")
                        if kind == "user":
                            role = "user"
                        elif kind == "reply":
                            role = "assistant"
                        else:
                            continue
                        text = (ev.get("text") or "").strip()
                        if not text:
                            continue
                        sid = ev.get("sid", "log")
                        idx = per_sid.get(sid, 0)
                        per_sid[sid] = idx + 1
                        ts = ev.get("ts")
                        try:
                            self.add(role, text, sid, idx,
                                     ts=(float(ts) if isinstance(ts, (int, float)) else None))
                            count += 1
                        except Exception:
                            pass
            except Exception as e:
                self._log(f"backfill {fp} failed: {e}")
        self._ensure_fts()
        self._log(f"backfilled {count} turns from {len(files)} log files")
        return count

    def warm_async(self, backfill: bool = True) -> None:
        """Load the model + open the store (and optionally backfill) off the hot path."""
        if not self._enabled:
            return
        def _w():
            try:
                if self._ensure() and backfill:
                    self.backfill_from_logs()
            except Exception:
                pass
        threading.Thread(target=_w, daemon=True).start()


# Module-level singleton (the runner uses one shared store).
_STORE: Optional[RecallStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> RecallStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = RecallStore()
    return _STORE


if __name__ == "__main__":  # tiny CLI for inspection
    import argparse
    ap = argparse.ArgumentParser(description="swarm-agent conversation recall")
    ap.add_argument("--search")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("-k", type=int, default=_TOPK)
    a = ap.parse_args()
    os.environ.setdefault("SWARM_RECALL_DEBUG", "1")
    s = get_store()
    if a.backfill:
        print("indexed:", s.backfill_from_logs())
    if a.search:
        for r in s.search(a.search, k=a.k):
            print(f"- ({r.get('role')}) {_clip(r.get('text',''), 160)}")
