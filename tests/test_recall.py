"""Recall store (LanceDB hybrid, CPU embeddings) — behavioural tests.

Skips cleanly if lancedb/fastembed aren't installed. Uses an explicit temp store +
enabled=True (conftest defaults SWARM_RECALL=0 for the rest of the suite)."""
import pytest

pytest.importorskip("lancedb")
pytest.importorskip("fastembed")

from swarm_agent.recall import RecallStore


@pytest.fixture()
def store(tmp_path):
    s = RecallStore(path=str(tmp_path / "recall.lance"), enabled=True)
    assert s._ensure(), "recall store failed to init (deps present but init errored)"
    return s


def _seed(s):
    turns = [
        ("user", "プロジェクトHermesのアクセスコードはBLUE-FALCON-7731。締め切りは14日。"),
        ("assistant", "了解、記録しました。"),
        ("user", "今日の天気は？"),
        ("assistant", "晴れです。"),
        ("user", "Pythonでフィボナッチを書いて"),
        ("assistant", "def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)"),
    ]
    for i, (r, t) in enumerate(turns):
        s.add(r, t, "sess", i)
    return turns


def test_far_past_fact_is_retrievable(store):
    _seed(store)
    rows = store.search("アクセスコードは何だった", k=3)
    joined = " ".join(r["text"] for r in rows)
    assert "BLUE-FALCON-7731" in joined


def test_cross_lingual_recall(store):
    _seed(store)
    # English query must surface the Japanese fact turn (multilingual embeddings).
    rows = store.search("what was the secret access code?", k=3)
    assert any("BLUE-FALCON-7731" in r["text"] for r in rows)


def test_block_excludes_recent_and_formats(store):
    turns = _seed(store)
    recent = [t for _, t in turns[-2:]]
    blk = store.block("アクセスコード教えて", exclude_texts=recent)
    assert "BLUE-FALCON-7731" in blk
    assert blk.startswith("[earlier messages from THIS conversation")
    # an excluded recent turn must not be re-injected
    assert "晴れです。" not in blk or "晴れ" not in recent[0]


def test_empty_store_returns_nothing(store):
    assert store.search("anything") == []
    assert store.block("anything") == ""


def test_disabled_store_is_noop(tmp_path):
    s = RecallStore(path=str(tmp_path / "x.lance"), enabled=False)
    s.add("user", "hi", "sess", 0)        # no-op, must not raise
    assert s.search("hi") == []
    assert s.block("hi") == ""


def test_parse_jst_window():
    import datetime as dt
    from swarm_agent.recall import parse_jst_window, _JST
    now = dt.datetime(2026, 6, 5, 15, 0, tzinfo=_JST)  # Friday
    assert parse_jst_window("普通の質問です", now=now) is None      # no temporal cue
    for term in ("昨日のこと", "先週の話", "今日やったこと", "2026-06-01の件", "6月3日に"):
        w = parse_jst_window(term, now=now)
        assert w is not None and w[0] < w[1], term
    # 先週 (Mon 5/25 .. Sun 5/31) must NOT include today (6/5)
    s, e = parse_jst_window("先週", now=now)
    today = dt.datetime(2026, 6, 5, 12, tzinfo=_JST).timestamp()
    assert not (s <= today < e)


def test_temporal_window_filters(store):
    import datetime as dt
    from swarm_agent.recall import _JST
    def ep(days_ago):
        d = (dt.datetime.now(_JST) - dt.timedelta(days=days_ago)).date()
        return dt.datetime(d.year, d.month, d.day, 12, tzinfo=_JST).timestamp()
    store.add("user", "先週の重要事項: トークンは APEX-55。", "s", 0, ts=ep(8))
    store.add("user", "今日の雑談、天気の話。", "s", 1, ts=ep(0))
    rows = store.search("先週の重要事項なんだっけ", k=3)
    joined = " ".join(r["text"] for r in rows)
    assert "APEX-55" in joined and "天気" not in joined
