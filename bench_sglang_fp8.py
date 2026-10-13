#!/usr/bin/env python3
"""Standalone sglang inference benchmark: bf16 vs fp8 (weight/kv/both), eager vs cuda-graph.

Mirrors the RL eval workload (AIME-2024, n=8, temp 0.7, max_new_tokens 8192) on a
single TP=1 server. Reports wall-time, total generated tokens, and decode tok/s.

Usage:
  python3 bench_sglang_fp8.py --mode bf16  --cuda-graph 1
  python3 bench_sglang_fp8.py --mode both  --cuda-graph 1   # weight+KV fp8
  python3 bench_sglang_fp8.py --mode kv    --cuda-graph 0   # KV-only fp8, eager
"""
import argparse
import time

from datasets import Dataset
from sglang.srt.entrypoints.engine import Engine

MODEL = "/jizhicfs/pkuhetu/siwei/model/Qwen2.5-Math-7B"
VAL = "/jizhicfs/pkuhetu/siwei/data/aime-2024.parquet"


def to_text(prompt) -> str:
    """AIME prompts are stored as OpenAI-style message lists; flatten to a string."""
    if isinstance(prompt, str):
        return prompt
    return "".join(m["content"] for m in prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "weight", "kv", "both"], default="bf16")
    ap.add_argument("--cuda-graph", type=int, default=1, help="0 = eager (disable cuda graphs)")
    ap.add_argument("--prompts", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    ap.add_argument("--warmup-n", type=int, default=2)
    ap.add_argument("--ignore-eos", type=int, default=0, help="force all seqs to generate exactly max_new tokens (clean speed test)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repeat", type=int, default=1, help="repeat the timed batch this many times")
    args = ap.parse_args()

    kw = dict(
        model_path=MODEL,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
        disable_cuda_graph=(args.cuda_graph == 0),
    )
    if args.mode in ("weight", "both"):
        kw["quantization"] = "fp8"
    if args.mode in ("kv", "both"):
        kw["kv_cache_dtype"] = "fp8_e4m3"

    ds = Dataset.from_parquet(VAL)
    prompts = [to_text(r["prompt"]) for r in ds][: args.prompts]
    sp = dict(max_new_tokens=args.max_new, n=args.n, temperature=args.temperature, top_p=1.0)
    if args.ignore_eos:
        sp["ignore_eos"] = True

    eng = Engine(**kw)
    try:
        # warmup so KV pool / cuda graphs settle before timing
        if args.warmup_n:
            eng.generate(prompt=prompts[: args.warmup_n], sampling_params=sp)

        times, totals = [], []
        for _ in range(args.repeat):
            t0 = time.time()
            outs = eng.generate(prompt=prompts, sampling_params=sp)
            dt = time.time() - t0
            gen_tokens = sum(o["meta_info"]["completion_tokens"] for o in outs)
            times.append(dt)
            totals.append(gen_tokens)
        best_dt = min(times)
        best_tok = totals[times.index(best_dt)]
        fin = {}
        for o in outs:
            fr = o["meta_info"].get("finish_reason", {})
            fin[fr.get("type", "?")] = fin.get(fr.get("type", "?"), 0) + 1
        avg_tok = best_tok / (len(outs) or 1)
        print(
            f"RESULT mode={args.mode} cuda_graph={args.cuda_graph} "
            f"prompts={args.prompts} n={args.n} max_new={args.max_new} "
            f"mem_fraction={args.mem_fraction} "
            f"wall={best_dt:.1f}s tokens={best_tok} seqs={len(outs)} "
            f"avg_tok/seq={avg_tok:.0f} finish={fin} "
            f"decode={best_tok / best_dt:.1f} tok/s "
            f"(best of {args.repeat})"
        )
    finally:
        eng.shutdown()


if __name__ == "__main__":
    main()
