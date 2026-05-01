#!/usr/bin/env python3
"""
Resumable experiment runner for the LLM moral-reasoning study.

Design:
- 5 model families × 2 modes = 10 configurations.
- Each (config × scenario × run) is one call.
- Results are appended to results/raw/{config_key}.jsonl IMMEDIATELY after every call.
- On relaunch, the runner re-reads each JSONL and SKIPS any (scenario_id, run_idx) tuple
  that already has a SUCCESSFUL record. Failed records (error or parse_error) are retried.
- Smart retries: transient (5xx, 429, network) → exponential backoff up to 3 attempts.
  Permanent (auth, billing, unsupported) → fail fast.
- Per-call timeout 90s.
- Parse-retry: if the model returns non-JSON, re-call the API once before giving up.
- Live cost tracker prints running spend estimate per provider.

Usage:
    python experiments/run_experiments.py                       # full plan
    python experiments/run_experiments.py --runs 1              # only N=1
    python experiments/run_experiments.py --categories tp mf    # subset
    python experiments/run_experiments.py --resume              # explicit resume (default behavior)
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import asdict
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Pre-import client modules to avoid concurrent first-import deadlocks
from experiments.models.anthropic_client import AnthropicClient  # noqa: E402
from experiments.models.openai_client import OpenAIClient        # noqa: E402
from experiments.models.gemini_client import GeminiClient        # noqa: E402
from experiments.models.together_client import TogetherClient    # noqa: E402

SCENARIOS_DIR = ROOT / "experiments" / "scenarios"
RESULTS_DIR = ROOT / "results" / "raw"

CATEGORY_FILES = {
    "tp": "trolley_problems.json",
    "mf": "moral_foundations.json",
    "pc": "paraphrase_consistency.json",
    "ds": "demographic_sensitivity.json",
    "cd": "contemporary_dilemmas.json",
}

# Each entry: (config_key, factory) where factory builds a fresh client
CONFIGS = [
    ("claude_instant",   lambda: AnthropicClient("claude-sonnet-4-6", mode="instant")),
    ("claude_light",     lambda: AnthropicClient("claude-sonnet-4-6", mode="light", effort="low")),

    ("gpt55_instant",    lambda: OpenAIClient("gpt-5.5", mode="instant")),
    ("gpt55_light",      lambda: OpenAIClient("gpt-5.5", mode="light")),

    ("gemini_instant",   lambda: GeminiClient("gemini-3-flash-preview", mode="instant")),
    ("gemini_light",     lambda: GeminiClient("gemini-3-flash-preview", mode="light")),

    ("deepseek_instant", lambda: TogetherClient("deepseek-ai/DeepSeek-V3.1", mode="instant")),
    ("deepseek_light",   lambda: TogetherClient("deepseek-ai/DeepSeek-V3.1", mode="light")),

    # Replaced Kimi K2.6 with Qwen3.5-397B (Alibaba flagship MoE) — apples-to-apples
    # tier with much faster inference on Together AI's serverless endpoint.
    ("qwen35_instant",   lambda: TogetherClient("Qwen/Qwen3.5-397B-A17B", mode="instant")),
    ("qwen35_light",     lambda: TogetherClient("Qwen/Qwen3.5-397B-A17B", mode="light")),
]

# Rough cost estimates per 1K tokens (input, output) for live cost tracking
PRICING = {
    "claude_instant":   (0.003, 0.015),
    "claude_light":     (0.003, 0.015),
    "gpt55_instant":    (0.00125, 0.010),
    "gpt55_light":      (0.00125, 0.010),
    "gemini_instant":   (0.0003, 0.0025),
    "gemini_light":     (0.0003, 0.0025),
    "deepseek_instant": (0.00060, 0.00170),
    "deepseek_light":   (0.00060, 0.00170),
    "qwen35_instant":   (0.00060, 0.00360),
    "qwen35_light":     (0.00060, 0.00360),
}

PERMANENT_ERROR_TOKENS = (
    "invalid_api_key", "unauthenticated", "401",
    "billing", "insufficient_quota", "exceeded your current quota",
    "model_not_found", "does not exist", "unsupported",
)
TRANSIENT_ERROR_TOKENS = (
    "rate", "quota_exceeded", "429",
    "5xx", "500", "502", "503", "504",
    "timeout", "timed out", "connection", "network",
)


# ─────────────────────────────────────────────────────────────────────────────

def load_scenarios(categories=None):
    cats = categories or list(CATEGORY_FILES.keys())
    out = []
    for cat in cats:
        with open(SCENARIOS_DIR / CATEGORY_FILES[cat]) as f:
            data = json.load(f)
        for s in data.get("scenarios", []):
            if "variant_a" in s:
                for key in ("variant_a", "variant_b"):
                    v = s[key]
                    out.append({
                        "id": v["id"],
                        "description": v["description"],
                        "question": v["question"],
                        "category": cat,
                        "base_id": s["id"],
                        "variant_key": key,
                    })
            elif "variants" in s:
                for v in s["variants"]:
                    out.append({
                        "id": v["id"],
                        "description": f"{s['description']} {v.get('demographic_detail', '')}".strip(),
                        "question": s["question"],
                        "category": cat,
                        "base_id": s["id"],
                        "demographic_type": v.get("demographic_type", ""),
                    })
            else:
                out.append({**s, "category": cat})
    return out


def load_completed_keys(out_file):
    """Read existing JSONL and return set of (scenario_id, run_idx) for SUCCESSFUL records."""
    completed = set()
    if not out_file.exists():
        return completed
    with open(out_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") or rec.get("parse_error"):
                continue
            if rec.get("judgment_binary") is None:
                continue
            sid = rec.get("scenario_meta", {}).get("id") or rec.get("scenario_id")
            run = rec.get("run", 0)
            if sid is not None:
                completed.add((sid, run))
    return completed


def classify_error(err_text: str) -> str:
    """Return 'permanent', 'transient', or 'unknown'."""
    if not err_text:
        return "unknown"
    low = err_text.lower()
    if any(t in low for t in PERMANENT_ERROR_TOKENS):
        return "permanent"
    if any(t in low for t in TRANSIENT_ERROR_TOKENS):
        return "transient"
    return "unknown"


def call_with_retries(client, scenario, max_attempts=3, base_delay=4.0, parse_retry=True):
    """Call the API with smart retries. Returns the (possibly failed) ModelResponse."""
    last = None
    for attempt in range(max_attempts):
        last = client.query_scenario(scenario)
        # Success path
        if not last.error and not last.parse_error and last.judgment_binary is not None:
            return last
        # Parse failure → one re-call (often a one-off truncation/format glitch)
        if not last.error and last.parse_error and parse_retry and attempt == 0:
            time.sleep(1.0)
            continue
        # API error: classify
        if last.error:
            kind = classify_error(last.error)
            if kind == "permanent":
                return last  # don't waste retries
            if kind == "transient":
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
                continue
            # unknown: try once more then give up
            if attempt < max_attempts - 1:
                time.sleep(base_delay)
                continue
        break
    return last


# ─────────────────────────────────────────────────────────────────────────────
# Live cost tracker (thread-safe)

class CostTracker:
    def __init__(self):
        self.lock = Lock()
        self.calls = {k: 0 for k, _ in CONFIGS}
        self.in_tokens = {k: 0 for k, _ in CONFIGS}
        self.out_tokens = {k: 0 for k, _ in CONFIGS}

    def record(self, config_key, in_tok, out_tok):
        with self.lock:
            self.calls[config_key] += 1
            self.in_tokens[config_key] += in_tok
            self.out_tokens[config_key] += out_tok

    def estimate_cost(self):
        total = 0.0
        per_config = {}
        for k in self.calls:
            in_p, out_p = PRICING[k]
            cost = (self.in_tokens[k] / 1000) * in_p + (self.out_tokens[k] / 1000) * out_p
            per_config[k] = cost
            total += cost
        return total, per_config

    def summary_line(self):
        total, _ = self.estimate_cost()
        n_calls = sum(self.calls.values())
        return f"[cost] {n_calls} calls, est. ${total:.2f} so far"


# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(prompt: str, response_text: str) -> tuple:
    """Rough char/4 estimate. Cheap and good enough for tracking."""
    return (len(prompt) // 4, max(1, len(response_text) // 4))


def run_one_config(config_key, factory, scenarios, n_runs, out_file, tracker, print_lock):
    """Run one (model × mode) configuration through all scenarios, all runs.
    Skip already-completed (scenario_id, run_idx) tuples. Append results immediately.
    """
    completed = load_completed_keys(out_file)
    n_skipped_init = len(completed)

    try:
        client = factory()
    except Exception as e:
        with print_lock:
            print(f"[{config_key}] FATAL: cannot build client: {e}")
        return

    target_calls = 0
    done_calls = 0

    for run_idx in range(n_runs):
        for scenario in scenarios:
            target_calls += 1
            sid = scenario["id"]
            if (sid, run_idx) in completed:
                continue

            t0 = time.time()
            response = call_with_retries(client, scenario)
            dt = time.time() - t0

            # Build record (note: using the original prompt text used inside the client's query_scenario)
            record = {
                "config_key": config_key,
                "run": run_idx,
                "model_id": getattr(client, "model_id", config_key),
                "mode": getattr(client, "mode", None),
                "scenario_meta": {k: scenario[k] for k in ("id", "category") if k in scenario},
                "scenario_id": sid,
                **asdict(response),
            }

            # Cost tracking (rough char-based)
            in_tok, out_tok = estimate_tokens(scenario["description"] + scenario["question"], response.raw_response or "")
            tracker.record(config_key, in_tok, out_tok)

            # Append to JSONL immediately (resumability is the priority)
            with open(out_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            done_calls += 1
            ok = (not response.error) and (not response.parse_error) and (response.judgment_binary is not None)
            status = "✓" if ok else f"✗ {(response.error or response.parse_error or '?')[:60]}"
            with print_lock:
                print(f"[{config_key:<18}] r{run_idx} {sid:<6} {status:<60} jb={response.judgment_binary} fw={response.primary_framework or '-':<14} {dt:.1f}s")

    with print_lock:
        skipped_total = n_skipped_init  # how many we skipped at start
        print(f"[{config_key}] done. attempted={done_calls}, skipped (resumed)={skipped_total}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", choices=list(CATEGORY_FILES.keys()))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--configs", nargs="+", default=[k for k, _ in CONFIGS],
                   choices=[k for k, _ in CONFIGS])
    p.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    # Verify required env vars
    needs = {
        "claude": "ANTHROPIC_API_KEY",
        "gpt55":  "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "deepseek": "TOGETHER_API_KEY",
        "qwen35": "TOGETHER_API_KEY",
    }
    missing = []
    for cfg in args.configs:
        family = cfg.split("_")[0]
        env = needs.get(family)
        if env and not os.environ.get(env):
            missing.append(f"{cfg} requires {env}")
    if missing:
        print("[FATAL] missing API keys:")
        for m in missing:
            print(f"  • {m}")
        sys.exit(1)

    scenarios = load_scenarios(args.categories)
    args.output.mkdir(parents=True, exist_ok=True)

    # Status banner
    print(f"\n{'=' * 80}")
    print(f"  LLM Moral-Reasoning Study — Full Run")
    print(f"  Configs:    {len(args.configs)}")
    print(f"  Scenarios:  {len(scenarios)}")
    print(f"  Runs/scen:  {args.runs}")
    print(f"  Total budget: {len(args.configs) * len(scenarios) * args.runs:,} calls")
    print(f"  Resume:     auto (skipping any already-successful records)")
    print(f"{'=' * 80}\n")

    # Pre-scan: count already-completed across all configs
    pre_scan = 0
    for cfg in args.configs:
        out_file = args.output / f"{cfg}.jsonl"
        pre_scan += len(load_completed_keys(out_file))
    if pre_scan > 0:
        print(f"  ⏯  Found {pre_scan} successful records on disk — these will be skipped.\n")

    tracker = CostTracker()
    print_lock = Lock()
    factories = {k: f for k, f in CONFIGS}

    with ThreadPoolExecutor(max_workers=len(args.configs)) as pool:
        futures = {
            pool.submit(
                run_one_config,
                cfg, factories[cfg], scenarios, args.runs,
                args.output / f"{cfg}.jsonl",
                tracker, print_lock,
            ): cfg
            for cfg in args.configs
        }
        # Periodic cost-tracker heartbeat thread? We just print on completion of each.
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[{cfg}] WORKER CRASHED: {e}")
                traceback.print_exc()
            print("  " + tracker.summary_line())

    total, per_cfg = tracker.estimate_cost()
    print(f"\n{'=' * 80}")
    print(f"  FINAL COST ESTIMATE: ${total:.2f}")
    for k in sorted(per_cfg.keys()):
        print(f"    {k:<20} ${per_cfg[k]:.2f}  ({tracker.calls[k]} calls)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
