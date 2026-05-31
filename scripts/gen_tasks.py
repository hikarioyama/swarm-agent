"""Generate realistic ~8K-token worker tasks for live fleet validation.

FLEET_OPTIMUM.md measured the C16-C64 operating region with UNIQUE ~8K-token context
per worker (prefix-cache off, worst case). To reproduce that operating point through the
REAL AIAgent path (not the synthetic httpx probe), we need independent single-turn tasks
whose prompt is ~8K tokens and whose answer is a few hundred tokens, with NO tool use
(so each task is one clean generation the DecodeGate can pin).

    PYTHONPATH=. python scripts/gen_tasks.py --n 64 --tokens 8000 --out examples/sweep_8k.jsonl

Each line: {"id","prompt","lane":"worker"}. Content is per-task-seed-varied so prompts are
byte-distinct (defeats trivial prefix-caching, matching the measurement's worst case).
"""
import argparse
import json
import random

# A pool of technical sentences; we shuffle + number them per task to hit a token target
# while keeping every task's context unique (like the measured worst case).
_SENTS = [
    "The cache line is the unit of coherence traffic between cores.",
    "Speculative decoding proposes K draft tokens the target verifies in one pass.",
    "Mixture-of-experts routes each token to a small subset of feed-forward experts.",
    "Paged KV cache stores attention keys and values in fixed-size blocks.",
    "Tensor parallelism shards each weight matrix across multiple GPUs.",
    "Prefix caching reuses the KV of a shared prompt across requests.",
    "Sliding-window attention bounds the KV that full layers would otherwise retain.",
    "An admission controller bounds the number of requests generating concurrently.",
    "Throughput rises with in-flight concurrency until the compute knee is reached.",
    "FP8 KV quantization halves cache footprint versus bf16 at small accuracy cost.",
    "Continuous batching admits new requests into the running batch each step.",
    "Preemption recomputes or swaps a request's KV when the cache is oversubscribed.",
    "Grouped-query attention shares key/value heads across query heads to shrink KV.",
    "The draft model's acceptance rate sets the realized speedup of speculation.",
    "Expert parallelism places different experts on different devices.",
    "A stigmergic queue lets workers coordinate through shared state, not messages.",
]


def make_prompt(seed: int, target_tokens: int) -> str:
    rng = random.Random(seed)
    # Calibrated against the live step3p7 tokenizer: this content (bracket markers +
    # technical terms) runs ~1.4 tokens/word, so ~0.53 words per target token.
    target_words = int(target_tokens * 0.53)
    parts = []
    n = 0
    i = 0
    while n < target_words:
        s = rng.choice(_SENTS)
        line = f"[ctx-{seed}-{i}] {s}"
        parts.append(line)
        n += len(line.split())
        i += 1
    context = "\n".join(parts)
    q = ("Using ONLY the numbered context above as flavour, write a single concise "
         "paragraph (about 120 words) explaining how bounding the number of concurrently "
         "generating requests keeps a GPU inference server near its throughput knee. "
         "Do not use any tools; answer directly.")
    return f"# Reference context (unique to this task)\n{context}\n\n# Question\n{q}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8000, help="approx prompt tokens per task")
    ap.add_argument("--lane", default="worker")
    ap.add_argument("--out", default="examples/sweep_8k.jsonl")
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for k in range(a.n):
            prompt = make_prompt(seed=1000 + k, target_tokens=a.tokens)
            f.write(json.dumps({"id": f"w{k:03d}", "prompt": prompt, "lane": a.lane},
                               ensure_ascii=False) + "\n")
    # rough token check on the last prompt
    approx = int(len(prompt.split()) / 0.75)
    print(f"wrote {a.n} tasks -> {a.out} (~{approx} tok/prompt, lane={a.lane})")


if __name__ == "__main__":
    main()
