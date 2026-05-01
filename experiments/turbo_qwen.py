#!/usr/bin/env python3
"""Parallel-within-config booster for qwen35_light to drain the last ~134 calls fast."""
import json, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from experiments.models.together_client import TogetherClient
from experiments.run_experiments import load_scenarios, call_with_retries

CONFIG = "qwen35_light"
MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
TARGET_N = 3
WORKERS = 3
OUT = ROOT / "results" / "raw" / f"{CONFIG}.jsonl"
write_lock = Lock()


def count_successes():
    counts = defaultdict(int)
    next_run = defaultdict(int)
    if not OUT.exists(): return counts, next_run
    for line in OUT.read_text().splitlines():
        if not line.strip(): continue
        try: r = json.loads(line)
        except: continue
        sid = r.get("scenario_id") or (r.get("scenario_meta") or {}).get("id")
        if sid is None: continue
        run = r.get("run", 0)
        next_run[sid] = max(next_run[sid], run + 1)
        if not r.get("error") and not r.get("parse_error") and r.get("judgment_binary") is not None:
            counts[sid] += 1
    return counts, next_run


def call_one(scenario, run_idx):
    client = TogetherClient(MODEL_ID, mode="light")
    response = call_with_retries(client, scenario, max_attempts=2, base_delay=3.0)
    record = {
        "config_key": CONFIG, "run": run_idx, "model_id": MODEL_ID, "mode": "light",
        "scenario_meta": {k: scenario[k] for k in ("id","category") if k in scenario},
        "scenario_id": scenario["id"], **asdict(response),
    }
    with write_lock:
        with open(OUT, "a") as f:
            f.write(json.dumps(record) + "\n")
    ok = (not response.error) and (not response.parse_error) and (response.judgment_binary is not None)
    return scenario["id"], run_idx, ok


def main():
    scenarios = {s["id"]: s for s in load_scenarios()}
    iter_n = 0
    while iter_n < 4:
        iter_n += 1
        counts, next_run = count_successes()
        deficit = {sid: TARGET_N - counts[sid] for sid in scenarios if counts[sid] < TARGET_N}
        if not deficit:
            print(f"\n✅ All scenarios at N≥{TARGET_N}.")
            return
        call_list = []
        for sid, need in deficit.items():
            rs = next_run[sid]
            for k in range(need):
                call_list.append((scenarios[sid], rs + k))
        print(f"=== iter {iter_n}: {len(deficit)} scenarios short, {len(call_list)} calls planned ===")
        t0 = time.time(); done = ok_n = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = [pool.submit(call_one, s, r) for s, r in call_list]
            for fut in as_completed(futs):
                done += 1
                sid, r, ok = fut.result()
                if ok: ok_n += 1
                rate = done/(time.time()-t0) if time.time()>t0 else 0
                print(f"  [{done}/{len(call_list)}] {sid} r{r} {'✓' if ok else '✗'}  rate={rate:.2f}/s ok={ok_n}/{done}")
        if ok_n < len(call_list)*0.3:
            print("Low success rate; stopping."); return


if __name__ == "__main__":
    if not os.environ.get("TOGETHER_API_KEY"):
        print("[FATAL] TOGETHER_API_KEY not set"); sys.exit(1)
    main()
