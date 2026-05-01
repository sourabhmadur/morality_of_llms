#!/usr/bin/env python3
"""
Top up kimi_light to full N=3 successful samples per scenario.
Skips scenarios already at N≥3. Uses parallel workers for speed.
"""
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from experiments.models.together_client import TogetherClient
from experiments.run_experiments import load_scenarios, call_with_retries

CONFIG = "kimi_light"
MODEL_ID = "moonshotai/Kimi-K2.6"
TARGET_N = 3
WORKERS = 6
OUT_FILE = ROOT / "results" / "raw" / f"{CONFIG}.jsonl"
write_lock = Lock()


def count_successes(out_file):
    """Return dict: scenario_id -> number of SUCCESSFUL records, plus next free run_idx."""
    counts = defaultdict(int)
    next_run = defaultdict(int)
    if not out_file.exists():
        return counts, next_run
    with open(out_file) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = r.get("scenario_id") or (r.get("scenario_meta") or {}).get("id")
            run = r.get("run", 0)
            if sid is None:
                continue
            next_run[sid] = max(next_run[sid], run + 1)
            if not r.get("error") and not r.get("parse_error") and r.get("judgment_binary") is not None:
                counts[sid] += 1
    return counts, next_run


def make_call(scenario, run_idx):
    client = TogetherClient(MODEL_ID, mode="light")
    response = call_with_retries(client, scenario, max_attempts=3, base_delay=4.0)
    record = {
        "config_key": CONFIG,
        "run": run_idx,
        "model_id": MODEL_ID,
        "mode": "light",
        "scenario_meta": {k: scenario[k] for k in ("id", "category") if k in scenario},
        "scenario_id": scenario["id"],
        **asdict(response),
    }
    with write_lock:
        with open(OUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    ok = (not response.error) and (not response.parse_error) and (response.judgment_binary is not None)
    return scenario["id"], run_idx, ok, response.error or response.parse_error or ""


def main():
    if not os.environ.get("TOGETHER_API_KEY"):
        print("[FATAL] TOGETHER_API_KEY not set"); sys.exit(1)

    scenarios = {s["id"]: s for s in load_scenarios()}

    # Iterate until every scenario has TARGET_N successful samples (or we give up)
    iteration = 0
    while True:
        iteration += 1
        counts, next_run = count_successes(OUT_FILE)
        deficit = {sid: TARGET_N - counts[sid] for sid in scenarios if counts[sid] < TARGET_N}
        if not deficit:
            print(f"\n✅ All {len(scenarios)} scenarios at N≥{TARGET_N}. Done.")
            return
        total_calls = sum(deficit.values())
        print(f"\n=== Iteration {iteration}: {len(deficit)} scenarios need top-up, {total_calls} calls ===")
        if iteration > 4:
            print("Stopping after 4 iterations to avoid infinite retry. Some scenarios may remain partial.")
            break

        # Build the call list
        call_list = []
        for sid, need in deficit.items():
            run_start = next_run[sid]
            for k in range(need):
                call_list.append((scenarios[sid], run_start + k))

        t0 = time.time()
        done = 0
        successes = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = [pool.submit(make_call, scen, run) for (scen, run) in call_list]
            for fut in as_completed(futs):
                done += 1
                sid, run, ok, err = fut.result()
                if ok: successes += 1
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                status = "✓" if ok else f"✗ {err[:50]}"
                print(f"  [{done}/{len(call_list)}] {sid} r{run} {status} | rate={rate:.2f}/s, ok={successes}/{done}")

        # If this iteration produced too few successes, bail to avoid wasting calls
        if successes < total_calls * 0.3:
            print(f"\nLow success rate ({successes}/{total_calls}). Stopping to avoid spend.")
            break

    # Final summary
    counts, _ = count_successes(OUT_FILE)
    short = [sid for sid in scenarios if counts[sid] < TARGET_N]
    if short:
        print(f"\n⚠️ {len(short)} scenarios still below N={TARGET_N}: {sorted(short)[:8]}{'...' if len(short)>8 else ''}")
    else:
        print(f"\n✅ All scenarios at N≥{TARGET_N}.")


if __name__ == "__main__":
    main()
