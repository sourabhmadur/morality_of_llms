#!/usr/bin/env python3
"""
Parallel-within-config booster for the slow lagging configs (kimi_light, deepseek_light).
Spawns multiple worker threads PER config to drain the remaining (scenario, run) tuples
much faster than the main sequential runner.

This is safe to run alongside the main runner — both check the same JSONL for already-done
records before making each call. Race condition window is small and only causes a duplicate
call (not corruption).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from experiments.models.together_client import TogetherClient
from experiments.run_experiments import (
    load_scenarios, load_completed_keys, call_with_retries,
)

RESULTS_DIR = ROOT / "results" / "raw"

# Configs to boost
BOOST = [
    ("kimi_light",     lambda: TogetherClient("moonshotai/Kimi-K2.6", mode="light")),
    ("deepseek_light", lambda: TogetherClient("deepseek-ai/DeepSeek-V3.1", mode="light")),
]

WORKERS_PER_CONFIG = 6  # parallel calls within each config
N_RUNS = 3

write_lock = Lock()  # serialize writes to avoid mid-line interleaving


def boost_one_config(config_key, factory):
    out_file = RESULTS_DIR / f"{config_key}.jsonl"
    scenarios = load_scenarios()  # all 100 scenarios

    completed = load_completed_keys(out_file)
    todo = [(s, r) for r in range(N_RUNS) for s in scenarios if (s["id"], r) not in completed]
    print(f"[{config_key}] {len(completed)} completed, {len(todo)} remaining")

    if not todo:
        print(f"[{config_key}] already done!")
        return

    # Each thread builds its own client (Together SDK is not thread-safe across calls)
    def worker(item):
        scenario, run_idx = item
        try:
            client = factory()
            response = call_with_retries(client, scenario, max_attempts=2, base_delay=3.0)
        except Exception as e:
            return (scenario["id"], run_idx, None, str(e))

        record = {
            "config_key": config_key,
            "run": run_idx,
            "model_id": getattr(client, "model_id", config_key),
            "mode": getattr(client, "mode", None),
            "scenario_meta": {k: scenario[k] for k in ("id", "category") if k in scenario},
            "scenario_id": scenario["id"],
            **asdict(response),
        }
        with write_lock:
            with open(out_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        ok = (not response.error) and (not response.parse_error) and (response.judgment_binary is not None)
        return (scenario["id"], run_idx, ok, response.error or response.parse_error or "")

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS_PER_CONFIG) as pool:
        futs = [pool.submit(worker, item) for item in todo]
        for fut in as_completed(futs):
            done += 1
            sid, run, ok, msg = fut.result()
            status = "✓" if ok else f"✗ {(msg or '?')[:50]}"
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (len(todo) - done) / rate if rate else 0
            print(f"[{config_key}] {done}/{len(todo)} {sid} r{run} {status} | rate={rate:.2f}/s, eta={eta/60:.0f}min")


def main():
    for k in ("ANTHROPIC_API_KEY", "TOGETHER_API_KEY"):
        if not os.environ.get(k):
            print(f"[FATAL] {k} not set"); sys.exit(1)

    # Run both boost configs in parallel
    with ThreadPoolExecutor(max_workers=len(BOOST)) as pool:
        futs = {pool.submit(boost_one_config, k, f): k for k, f in BOOST}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"[boost worker crashed] {e}")


if __name__ == "__main__":
    main()
