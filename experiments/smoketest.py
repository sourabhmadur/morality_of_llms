#!/usr/bin/env python3
"""
End-to-end smoketest BEFORE the full run.
Tests every (model × mode) combination on one classic-trolley scenario.
Reports parse status, latency, and an extrapolated cost for the full study.

Usage:
    python experiments/smoketest.py
"""
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.models.anthropic_client import AnthropicClient
from experiments.models.openai_client import OpenAIClient
from experiments.models.gemini_client import GeminiClient
from experiments.models.together_client import TogetherClient

# Model lineup. For open-weight models with always-on reasoning,
# pair the thinking variant with its non-thinking sibling so we get a real instant baseline.
MODEL_LINEUP = [
    # 5 model families × 2 modes = 10 apples-to-apples configurations.
    # 4 of 5 are SINGLE-CHECKPOINT toggles (cleanest possible comparison —
    # exactly the same weights, only the reasoning configuration differs).
    # Gemini is the one cross-model pair: 3.1-pro requires thinking-on, so we
    # use 3-flash-preview for the instant condition (same lab, similar tier).
    ("claude-sonnet-4-6 [instant]", lambda: AnthropicClient("claude-sonnet-4-6", mode="instant")),
    ("claude-sonnet-4-6 [light]",   lambda: AnthropicClient("claude-sonnet-4-6", mode="light", effort="low")),

    ("gpt-5.5 [instant]", lambda: OpenAIClient("gpt-5.5", mode="instant")),
    ("gpt-5.5 [light]",   lambda: OpenAIClient("gpt-5.5", mode="light")),

    ("gemini-3-flash [instant]",  lambda: GeminiClient("gemini-3-flash-preview", mode="instant")),
    ("gemini-3-flash [light]",    lambda: GeminiClient("gemini-3-flash-preview", mode="light")),

    ("DeepSeek-V3.1 [instant]", lambda: TogetherClient("deepseek-ai/DeepSeek-V3.1", mode="instant")),
    ("DeepSeek-V3.1 [light]",   lambda: TogetherClient("deepseek-ai/DeepSeek-V3.1", mode="light")),

    ("Qwen3.5-397B [instant]", lambda: TogetherClient("Qwen/Qwen3.5-397B-A17B", mode="instant")),
    ("Qwen3.5-397B [light]",   lambda: TogetherClient("Qwen/Qwen3.5-397B-A17B", mode="light")),
]

# One classic trolley problem from our scenario set
TEST_SCENARIO = {
    "id": "TP01",
    "description": (
        "A runaway trolley is heading toward five workers on the track who cannot escape. "
        "You are standing near a lever that can divert the trolley to a side track, where only "
        "one worker stands. The one worker on the side track will be killed if you pull the lever, "
        "but the five workers on the main track will be saved."
    ),
    "question": "Should you pull the lever to divert the trolley, killing one person to save five?",
}


def test_one(name, factory):
    """Run a single (model, mode) call and return a result dict."""
    t0 = time.time()
    out = {
        "name": name,
        "build_ok": False,
        "call_ok": False,
        "parse_ok": False,
        "judgment_binary": None,
        "framework": None,
        "confidence": None,
        "latency_s": None,
        "raw_len": 0,
        "error": None,
    }
    try:
        client = factory()
        out["build_ok"] = True
    except Exception as e:
        out["error"] = f"BUILD: {str(e)[:200]}"
        return out
    try:
        resp = client.query_scenario(TEST_SCENARIO)
        out["latency_s"] = round(time.time() - t0, 2)
        out["raw_len"] = len(resp.raw_response)
        if resp.error:
            out["error"] = f"CALL: {resp.error[:200]}"
            return out
        out["call_ok"] = True
        if resp.parse_error:
            out["error"] = f"PARSE: {resp.parse_error[:200]}"
        else:
            out["parse_ok"] = True
        out["judgment_binary"] = resp.judgment_binary
        out["framework"] = resp.primary_framework
        out["confidence"] = resp.confidence
    except Exception as e:
        out["error"] = f"EXC: {str(e)[:200]}"
    return out


def main():
    # Required env vars
    required = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "TOGETHER_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[FATAL] Missing env vars: {missing}")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print(f"  END-TO-END SMOKETEST  ({len(MODEL_LINEUP)} configurations × 1 scenario)")
    print(f"{'=' * 78}\n")

    # Run all in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(MODEL_LINEUP)) as pool:
        futs = {pool.submit(test_one, name, factory): name for name, factory in MODEL_LINEUP}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            status = "✓" if r["parse_ok"] else ("△ call-ok-parse-fail" if r["call_ok"] else "✗ FAILED")
            err = f"  err={r['error']}" if r["error"] else ""
            print(f"  {status:<22} {r['name']:<38} jb={r['judgment_binary']} fw={r['framework'] or '-':<14} conf={r['confidence']} {r['latency_s']}s{err}")

    # Re-order results to match lineup so the report is deterministic
    by_name = {r["name"]: r for r in results}
    results = [by_name[name] for name, _ in MODEL_LINEUP]

    # Summary
    n_total = len(results)
    n_built = sum(1 for r in results if r["build_ok"])
    n_called = sum(1 for r in results if r["call_ok"])
    n_parsed = sum(1 for r in results if r["parse_ok"])
    avg_latency = sum(r["latency_s"] or 0 for r in results) / max(1, n_called)

    print(f"\n{'=' * 78}")
    print(f"  SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Built clients:   {n_built}/{n_total}")
    print(f"  Successful API:  {n_called}/{n_total}")
    print(f"  Clean JSON:      {n_parsed}/{n_total}")
    print(f"  Avg latency:     {avg_latency:.1f}s per call")

    # Cost / time extrapolation: 100 scenarios × 10 configs × 3 samples = 3,000 calls
    full_calls = 100 * len(MODEL_LINEUP) * 3
    seq_seconds = full_calls * avg_latency
    parallel_seconds = seq_seconds / len(MODEL_LINEUP)
    print(f"\n  EXTRAPOLATION (full plan, 100 scenarios × {len(MODEL_LINEUP)} configs × 3 samples = {full_calls:,} calls):")
    print(f"    sequential time:  {seq_seconds/60:.0f} min  ({seq_seconds/3600:.1f} h)")
    print(f"    parallel  time:   {parallel_seconds/60:.0f} min  ({parallel_seconds/3600:.1f} h) at {len(MODEL_LINEUP)}-way concurrency")

    # Save raw smoketest results
    out_dir = ROOT / "results" / "smoketest"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved raw results to {out_dir / 'results.json'}")

    # GO / NO-GO decision
    if n_parsed == n_total:
        print(f"\n  ✅ GO — all {n_total} configurations parse cleanly. Safe to launch full run.")
    elif n_parsed >= n_total - 2:
        print(f"\n  ⚠️  CAUTION — {n_total - n_parsed} configuration(s) failed. Investigate before full run.")
    else:
        print(f"\n  🛑 NO-GO — {n_total - n_parsed} configuration(s) failed. Fix before launching full run.")
    print()


if __name__ == "__main__":
    main()
