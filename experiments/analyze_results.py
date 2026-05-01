#!/usr/bin/env python3
"""
Analysis pipeline for the LLM moral-reasoning study.

Reads all results/raw/*.jsonl files, computes 7 metrics, and emits:
  - results/processed/*.csv (machine-readable summaries)
  - paper/figures/*.pdf     (figures for the manuscript)
  - paper/tables/*.tex      (auto-generated LaTeX tables)

Metrics:
  M1. Framework distribution per (model, mode)
  M2. Cross-model agreement (Krippendorff's alpha + pairwise Cohen's kappa)
  M3. Paraphrase consistency rate within each (model, mode)
  M4. Demographic bias coefficient
  M5. Hard-case entropy per scenario
  M6. Confidence calibration by category and framework
  M7. Effect of thinking — verdict-flip rate, framework-shift rate, paraphrase-Δ, demographic-Δ

Usage: python experiments/analyze_results.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "results" / "raw"
PROC_DIR = ROOT / "results" / "processed"
FIG_DIR = ROOT / "paper" / "figures"
TBL_DIR = ROOT / "paper" / "tables"
SCEN_DIR = ROOT / "experiments" / "scenarios"

# Display ordering for figures
CONFIG_ORDER = [
    "claude_instant", "claude_light",
    "gpt55_instant",  "gpt55_light",
    "gemini_instant", "gemini_light",
    "deepseek_instant", "deepseek_light",
    "qwen35_instant",   "qwen35_light",
]
MODEL_FAMILIES = ["claude", "gpt55", "gemini", "deepseek", "qwen35"]
MODEL_PRETTY = {
    "claude": "Claude Sonnet 4.6",
    "gpt55": "GPT-5.5",
    "gemini": "Gemini 3 Flash",
    "deepseek": "DeepSeek-V3.1",
    "qwen35": "Qwen3.5-397B",
}
MODE_PRETTY = {"instant": "Instant", "light": "Thinking"}
FRAMEWORK_ORDER = ["utilitarian", "deontological", "virtue_ethics",
                   "care_ethics", "contractualist", "other"]
FW_PRETTY = {
    "utilitarian": "Utilitarian", "deontological": "Deontological",
    "virtue_ethics": "Virtue", "care_ethics": "Care",
    "contractualist": "Contractualist", "other": "Other",
}
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


# ─── Loading ────────────────────────────────────────────────────────────────

def load_all_results() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(r)
    if not rows:
        raise SystemExit(f"No data in {RAW_DIR}. Run run_experiments.py first.")
    df = pd.DataFrame(rows)
    df["family"] = df["config_key"].str.split("_").str[0]
    df["mode"] = df["config_key"].str.split("_").str[1]
    df["scenario_id"] = df.apply(
        lambda r: r.get("scenario_id") or (r.get("scenario_meta") or {}).get("id"),
        axis=1
    )
    df["category"] = df.apply(
        lambda r: (r.get("scenario_meta") or {}).get("category", ""),
        axis=1
    )
    df["ok"] = df["error"].isna() & df["parse_error"].isna() & df["judgment_binary"].notna()
    return df


def load_scenario_metadata() -> pd.DataFrame:
    """For paraphrase pairs and demographic triplets we need base_id and variant info."""
    rows = []
    for fname, cat in [
        ("trolley_problems.json", "tp"),
        ("moral_foundations.json", "mf"),
        ("paraphrase_consistency.json", "pc"),
        ("demographic_sensitivity.json", "ds"),
        ("contemporary_dilemmas.json", "cd"),
    ]:
        with open(SCEN_DIR / fname) as f:
            data = json.load(f)
        for s in data.get("scenarios", []):
            if "variant_a" in s:
                for k in ("variant_a", "variant_b"):
                    rows.append({"scenario_id": s[k]["id"], "category": cat,
                                 "base_id": s["id"], "variant_key": k})
            elif "variants" in s:
                for v in s["variants"]:
                    rows.append({"scenario_id": v["id"], "category": cat,
                                 "base_id": s["id"], "demographic_type": v.get("demographic_type", "")})
            else:
                rows.append({"scenario_id": s["id"], "category": cat, "base_id": s["id"]})
    return pd.DataFrame(rows)


# ─── M1: Framework distribution ─────────────────────────────────────────────

def m1_framework_distribution(df: pd.DataFrame) -> pd.DataFrame:
    g = (df[df["ok"]]
         .groupby(["config_key", "primary_framework"])
         .size()
         .unstack(fill_value=0))
    for fw in FRAMEWORK_ORDER:
        if fw not in g.columns:
            g[fw] = 0
    g = g[FRAMEWORK_ORDER]
    g = g.div(g.sum(axis=1), axis=0)  # normalize to fractions
    g = g.reindex([c for c in CONFIG_ORDER if c in g.index])
    g.to_csv(PROC_DIR / "m1_framework_distribution.csv")
    plot_m1(g)
    return g


def plot_m1(g: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    g.plot(kind="bar", stacked=True, ax=ax,
           color=sns.color_palette("Set2", n_colors=len(FRAMEWORK_ORDER)))
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Fraction of responses")
    ax.set_title("Self-labeled ethical framework by configuration")
    pretty_xticks = []
    for t in ax.get_xticklabels():
        cfg = t.get_text()
        fam, mode = cfg.split("_")
        pretty_xticks.append(f"{MODEL_PRETTY.get(fam, fam)}\n({mode})")
    ax.set_xticklabels(pretty_xticks, rotation=20, ha="right", fontsize=8)
    ax.legend([FW_PRETTY[fw] for fw in FRAMEWORK_ORDER],
              loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_framework_distribution.pdf", bbox_inches="tight")
    plt.close()


# ─── M2: Cross-model agreement (Krippendorff alpha + pairwise kappa) ────────

def cohen_kappa(a: list, b: list) -> float:
    """Compute Cohen's kappa for binary labels (assumes equal-length lists)."""
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p1a = sum(a) / n; p1b = sum(b) / n
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=95, seed=42):
    """Return (point_estimate, lower, upper) for stat_fn over `values`."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots[i] = stat_fn(sample)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return float(stat_fn(arr)), float(np.percentile(boots, lo)), float(np.percentile(boots, hi))


def bootstrap_alpha(rater_matrix, n_boot=1000, seed=42):
    """Bootstrap CI for Krippendorff's alpha by resampling scenarios (columns)."""
    try:
        import krippendorff
    except ImportError:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_items = rater_matrix.shape[1]
    if n_items < 2:
        return float("nan"), float("nan"), float("nan")
    point = krippendorff.alpha(reliability_data=rater_matrix, level_of_measurement="nominal")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        m = rater_matrix[:, idx]
        # skip degenerate resamples (e.g., all-same column)
        try:
            a = krippendorff.alpha(reliability_data=m, level_of_measurement="nominal")
            if not np.isnan(a):
                boots.append(a)
        except Exception:
            pass
    if not boots:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def m2_cross_model_agreement(df: pd.DataFrame) -> dict:
    out = {}
    for mode in ["instant", "light"]:
        sub = df[(df["mode"] == mode) & (df["ok"])]
        # majority vote per (config_key, scenario_id)
        piv = (sub.groupby(["config_key", "scenario_id"])["judgment_binary"]
                  .agg(lambda x: int(round(x.mean())))
                  .unstack("config_key"))
        # only scenarios where ALL configs in this mode have an answer
        configs_in_mode = [c for c in CONFIG_ORDER if c.endswith(mode) and c in piv.columns]
        if len(configs_in_mode) < 2:
            continue
        piv = piv[configs_in_mode].dropna()

        # Krippendorff's alpha with bootstrap CI
        alpha, alpha_lo, alpha_hi = bootstrap_alpha(piv.T.values)

        # Pairwise kappa matrix
        n = len(configs_in_mode)
        kappa = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                a = piv.iloc[:, i].astype(int).tolist()
                b = piv.iloc[:, j].astype(int).tolist()
                kappa[i, j] = cohen_kappa(a, b)

        kappa_df = pd.DataFrame(kappa, index=configs_in_mode, columns=configs_in_mode)
        kappa_df.to_csv(PROC_DIR / f"m2_kappa_{mode}.csv")

        out[mode] = {"alpha": alpha, "alpha_ci": (alpha_lo, alpha_hi),
                     "kappa": kappa_df, "n_scenarios": len(piv), "rater_matrix": piv}

        # Heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(kappa_df, annot=True, fmt=".2f", vmin=-0.2, vmax=1.0,
                    cmap="RdYlGn", center=0.5, ax=ax, cbar_kws={"label": "Cohen's κ"})
        ax.set_title(f"Pairwise binary-judgment agreement — {MODE_PRETTY[mode]} mode\n"
                     f"Krippendorff's α = {alpha:.3f} [{alpha_lo:.3f}, {alpha_hi:.3f}], n={len(piv)} scenarios")
        pretty_labels = [MODEL_PRETTY.get(c.split("_")[0], c.split("_")[0]) for c in configs_in_mode]
        ax.set_xticklabels(pretty_labels, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(pretty_labels, rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"fig2_kappa_{mode}.pdf", bbox_inches="tight")
        plt.close()
    return out


def m2b_easy_vs_hard_alpha(df: pd.DataFrame, m2: dict) -> dict:
    """Test H4: cross-model agreement on easy vs hard scenarios.
    Easy = all 5 models agree in instant mode; hard = at least 1 disagrees."""
    if "instant" not in m2:
        return {}
    instant_piv = m2["instant"]["rater_matrix"]   # rows=scenarios, cols=configs
    # variance over rows: 0 = unanimous, >0 = some disagreement
    row_var = instant_piv.var(axis=1)
    easy_ids = instant_piv[row_var == 0].index.tolist()
    hard_ids = instant_piv[row_var > 0].index.tolist()
    out = {"easy": {"n": len(easy_ids)}, "hard": {"n": len(hard_ids)}}
    for mode in ["instant", "light"]:
        if mode not in m2: continue
        piv = m2[mode]["rater_matrix"]
        for label, ids in [("easy", easy_ids), ("hard", hard_ids)]:
            mask = piv.index.isin(ids)
            sub = piv[mask]
            if len(sub) < 2:
                out[label][mode] = (float("nan"), float("nan"), float("nan"))
                continue
            a, lo, hi = bootstrap_alpha(sub.T.values)
            out[label][mode] = (a, lo, hi)
    # Save as CSV
    rows = []
    for diff in ("easy", "hard"):
        n = out[diff]["n"]
        for mode in ("instant", "light"):
            if mode in out[diff]:
                a, lo, hi = out[diff][mode]
                rows.append({"difficulty": diff, "mode": mode, "n_scenarios": n,
                             "alpha": a, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(PROC_DIR / "m2b_easy_vs_hard.csv", index=False)
    return out


# ─── M3: Paraphrase consistency rate ────────────────────────────────────────

def m3_paraphrase_consistency(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    pc = meta[meta["category"] == "pc"].copy()
    if pc.empty:
        return pd.DataFrame()
    # majority binary judgment per (config_key, scenario_id)
    j = (df[df["ok"]]
         .groupby(["config_key", "scenario_id"])["judgment_binary"]
         .agg(lambda x: int(round(x.mean()))))
    out = []
    for cfg in CONFIG_ORDER:
        # for each base_id, collect a list of {0,1} indicators (1 if pair consistent)
        flags = []
        for base_id, group in pc.groupby("base_id"):
            ids = group["scenario_id"].tolist()
            if len(ids) < 2:
                continue
            try:
                a = j.loc[(cfg, ids[0])]
                b = j.loc[(cfg, ids[1])]
                flags.append(1 if a == b else 0)
            except KeyError:
                continue
        rate, lo, hi = bootstrap_ci(flags) if flags else (float("nan"),)*3
        out.append({"config_key": cfg, "consistent": sum(flags), "total": len(flags),
                    "rate": rate, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(out)
    res.to_csv(PROC_DIR / "m3_paraphrase_consistency.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    families = [MODEL_PRETTY.get(c.split("_")[0], c.split("_")[0]) for c in res["config_key"]]
    modes = [c.split("_")[1] for c in res["config_key"]]
    plot_df = pd.DataFrame({"family": families, "mode": modes, "rate": res["rate"]})
    sns.barplot(data=plot_df, x="family", y="rate", hue="mode",
                hue_order=["instant", "light"], palette="Set2", ax=ax,
                order=[MODEL_PRETTY[f] for f in MODEL_FAMILIES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Paraphrase consistency rate")
    ax.set_xlabel("Model family")
    ax.set_title("M3: Paraphrase Consistency Rate (higher = more consistent)")
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_paraphrase_consistency.pdf", bbox_inches="tight")
    plt.close()
    return res


# ─── M4: Demographic bias coefficient ──────────────────────────────────────

def m4_demographic_bias(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ds = meta[meta["category"] == "ds"].copy()
    if ds.empty:
        return pd.DataFrame()
    j = (df[df["ok"]]
         .groupby(["config_key", "scenario_id"])["judgment_binary"]
         .agg(lambda x: int(round(x.mean()))))
    out = []
    directional_rows = []  # for SG-4: which variant was the outlier?
    for cfg in CONFIG_ORDER:
        flags = []
        for base_id, group in ds.groupby("base_id"):
            ids = group["scenario_id"].tolist()
            demo_types = group["demographic_type"].tolist() if "demographic_type" in group else [""]*len(ids)
            if len(ids) < 2:
                continue
            try:
                vals = [j.loc[(cfg, sid)] for sid in ids]
                is_biased = (len(set(vals)) > 1)
                flags.append(1 if is_biased else 0)
                if is_biased:
                    # which variant disagrees with the majority?
                    from collections import Counter as _C
                    majority = _C(vals).most_common(1)[0][0]
                    for sid, dt, v in zip(ids, demo_types, vals):
                        if v != majority:
                            directional_rows.append({
                                "config_key": cfg, "base_id": base_id,
                                "outlier_scenario": sid, "outlier_demo_type": dt,
                                "outlier_verdict": v, "majority_verdict": majority,
                            })
            except KeyError:
                continue
        rate, lo, hi = bootstrap_ci(flags) if flags else (float("nan"),)*3
        out.append({"config_key": cfg, "n_biased_triplets": sum(flags), "n_total": len(flags),
                    "bias_coefficient": rate, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(out)
    res.to_csv(PROC_DIR / "m4_demographic_bias.csv", index=False)
    if directional_rows:
        pd.DataFrame(directional_rows).to_csv(PROC_DIR / "m4b_directional_bias.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    families = [MODEL_PRETTY.get(c.split("_")[0], c.split("_")[0]) for c in res["config_key"]]
    modes = [c.split("_")[1] for c in res["config_key"]]
    plot_df = pd.DataFrame({"family": families, "mode": modes, "bias": res["bias_coefficient"]})
    sns.barplot(data=plot_df, x="family", y="bias", hue="mode",
                hue_order=["instant", "light"], palette="Set2", ax=ax,
                order=[MODEL_PRETTY[f] for f in MODEL_FAMILIES])
    ax.set_ylabel("Demographic-judgment inconsistency")
    ax.set_xlabel("Model family")
    ax.set_title("M4: Demographic Inconsistency (lower = more demographically blind)")
    ax.set_ylim(0, max(0.6, plot_df["bias"].max() * 1.2 if not plot_df["bias"].isna().all() else 1.0))
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_demographic_bias.pdf", bbox_inches="tight")
    plt.close()
    return res


# ─── M5: Hard-case entropy per scenario ────────────────────────────────────

def m5_hard_case_entropy(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mode in ["instant", "light"]:
        sub = df[(df["mode"] == mode) & (df["ok"])]
        # for each scenario, get the binary judgments across model families
        piv = (sub.groupby(["family", "scenario_id"])["judgment_binary"]
                  .agg(lambda x: int(round(x.mean())))
                  .unstack("family"))
        for sid, row in piv.iterrows():
            vals = row.dropna().astype(int).tolist()
            if len(vals) < 2:
                continue
            p1 = sum(vals) / len(vals)
            if p1 in (0.0, 1.0):
                ent = 0.0
            else:
                ent = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
            out.append({"mode": mode, "scenario_id": sid, "entropy": ent,
                        "n_models": len(vals), "p1": p1})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res.to_csv(PROC_DIR / "m5_hard_case_entropy.csv", index=False)
    return res


# ─── M6: Confidence calibration ────────────────────────────────────────────

def m_trace_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """SG-1: Validate self-labeled framework against actual trace content.

    For each thinking-mode trace, count keyword hits per ethical framework and
    test whether the trace's dominant framework matches the self-labeled one.
    """
    if "reasoning_trace" not in df.columns:
        return pd.DataFrame()
    sub = df[df["ok"] & (df["mode"] == "light") & df["reasoning_trace"].notna() &
             (df["reasoning_trace"].str.len() > 50)].copy()
    if sub.empty:
        return pd.DataFrame()

    # Keyword cues per framework. Conservative — only count strong indicators.
    cues = {
        "utilitarian": [
            "utilitarian", "consequentialist", "consequentialism",
            "greatest good", "minimize harm", "maximize", "net benefit",
            "outcome", "aggregate", "overall good", "more lives",
        ],
        "deontological": [
            "deontolog", "kantian", "categorical imperative", "rights-based",
            "mere means", "intrinsically wrong", "duty", "dignity",
            "use as a means", "treat them as", "absolute prohibition",
            "regardless of consequences",
        ],
        "virtue_ethics": [
            "virtue", "character", "what a virtuous", "phron",
            "courage", "wisdom",
        ],
        "care_ethics": [
            "care ethic", "relationship", "vulnerab", "dependents",
            "compassion",
        ],
        "contractualist": [
            "contractuali", "social contract", "rawls", "veil of ignorance",
            "principles all could", "reasonable people would agree",
        ],
    }

    rows = []
    for _, r in sub.iterrows():
        trace = r["reasoning_trace"].lower()
        scores = {fw: sum(trace.count(c) for c in clist) for fw, clist in cues.items()}
        # if the trace is very short or has no cues, mark as "indeterminate"
        if max(scores.values()) == 0:
            trace_top = None
        else:
            trace_top = max(scores, key=scores.get)
        rows.append({
            "config_key": r["config_key"], "family": r["family"],
            "scenario_id": r["scenario_id"], "self_label": r["primary_framework"],
            "trace_top": trace_top, "trace_len": len(r["reasoning_trace"]),
            **{f"score_{k}": v for k, v in scores.items()},
        })
    res = pd.DataFrame(rows)

    # Concordance summary: what fraction of thinking-mode traces have the
    # trace-dominant framework match the self-label?
    summary = []
    for fam, grp in res.groupby("family"):
        valid = grp[grp["trace_top"].notna()]
        if len(valid) == 0:
            summary.append({"family": fam, "n_with_cues": 0,
                            "n_total": len(grp), "concordance": float("nan"),
                            "ci_lo": float("nan"), "ci_hi": float("nan")})
            continue
        flags = (valid["trace_top"] == valid["self_label"]).astype(int).tolist()
        rate, lo, hi = bootstrap_ci(flags)
        summary.append({"family": fam, "n_with_cues": len(valid),
                        "n_total": len(grp), "concordance": rate,
                        "ci_lo": lo, "ci_hi": hi})
    summary_df = pd.DataFrame(summary)
    res.to_csv(PROC_DIR / "m_trace_concordance_per_call.csv", index=False)
    summary_df.to_csv(PROC_DIR / "m_trace_concordance_summary.csv", index=False)

    # LaTeX table
    if not summary_df.empty:
        lines = ["\\begin{tabular}{lrrr}", "\\toprule",
                 "Model family & Traces with cues / total & Concordance & 95\\% CI \\\\",
                 "\\midrule"]
        for _, r in summary_df.iterrows():
            fam_p = MODEL_PRETTY.get(r["family"], r["family"])
            n_cue = int(r["n_with_cues"]); n_tot = int(r["n_total"])
            if pd.isna(r["concordance"]):
                lines.append(f"{fam_p} & {n_cue} / {n_tot} & --- & --- \\\\")
            else:
                lines.append(f"{fam_p} & {n_cue} / {n_tot} & "
                             f"{r['concordance']:.2f} & "
                             f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TBL_DIR / "tab_trace_concordance.tex").write_text("\n".join(lines))
    return summary_df


def m_reasoning_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Reasoning-tokens used per (model family, mode) — to document the apples-to-apples
    asymmetry across providers."""
    if "reasoning_tokens" not in df.columns:
        return pd.DataFrame()
    sub = df[df["ok"] & df["reasoning_tokens"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    g = (sub.groupby(["family", "mode"])["reasoning_tokens"]
            .agg(["mean", "std", "count"])
            .reset_index())
    g.to_csv(PROC_DIR / "m_reasoning_tokens.csv", index=False)

    # Build a LaTeX table summarizing the apples-to-apples settings
    light_only = g[g["mode"] == "light"].sort_values("family")
    if not light_only.empty:
        lines = ["\\begin{tabular}{lrr}", "\\toprule",
                 "Model family & Mean reasoning tokens & Std \\\\", "\\midrule"]
        for _, r in light_only.iterrows():
            lines.append(f"{MODEL_PRETTY.get(r['family'], r['family'])} & "
                         f"{r['mean']:.0f} & {r['std']:.0f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TBL_DIR / "tab_reasoning_tokens.tex").write_text("\n".join(lines))
    return g


def m6_confidence(df: pd.DataFrame) -> pd.DataFrame:
    s = (df[df["ok"]]
         .groupby(["config_key", "category"])["confidence"]
         .mean()
         .reset_index())
    s.to_csv(PROC_DIR / "m6_confidence.csv", index=False)
    return s


# ─── M7: Effect of thinking (the headline) ─────────────────────────────────

def m7_thinking_effect(df: pd.DataFrame) -> dict:
    """For each model family and scenario, compare instant vs light judgments."""
    j_per_run = df[df["ok"]].copy()
    # collapse runs by majority vote per (family, mode, scenario)
    maj = (j_per_run.groupby(["family", "mode", "scenario_id"])
                    .agg(jb=("judgment_binary", lambda x: int(round(x.mean()))),
                         fw=("primary_framework", lambda x: x.mode().iloc[0] if not x.mode().empty else None))
                    .reset_index())
    out_rows = []
    for fam, group in maj.groupby("family"):
        piv_jb = group.pivot(index="scenario_id", columns="mode", values="jb").dropna()
        piv_fw = group.pivot(index="scenario_id", columns="mode", values="fw").dropna()
        if "instant" not in piv_jb.columns or "light" not in piv_jb.columns:
            continue
        # bootstrap over scenarios
        flip_flags = (piv_jb["instant"] != piv_jb["light"]).astype(int).tolist()
        shift_flags = (piv_fw["instant"] != piv_fw["light"]).astype(int).tolist()
        v_rate, v_lo, v_hi = bootstrap_ci(flip_flags) if flip_flags else (float("nan"),)*3
        f_rate, f_lo, f_hi = bootstrap_ci(shift_flags) if shift_flags else (float("nan"),)*3
        out_rows.append({"family": fam,
                         "verdict_flip_rate": v_rate, "verdict_ci_lo": v_lo, "verdict_ci_hi": v_hi,
                         "framework_shift_rate": f_rate, "fw_ci_lo": f_lo, "fw_ci_hi": f_hi,
                         "n_scenarios": len(piv_jb)})

    res = pd.DataFrame(out_rows)
    res.to_csv(PROC_DIR / "m7_thinking_effect.csv", index=False)

    if not res.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(res))
        w = 0.35
        ax.bar(x - w/2, res["verdict_flip_rate"], w, label="Verdict-flip rate", color=sns.color_palette("Set2")[0])
        ax.bar(x + w/2, res["framework_shift_rate"], w, label="Framework-shift rate", color=sns.color_palette("Set2")[1])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_PRETTY[f] for f in res["family"]], rotation=15, ha="right")
        ax.set_ylabel("Rate (instant ≠ thinking)")
        ax.set_title("M7: Effect of Thinking on Moral Judgment (per model family)")
        ax.set_ylim(0, max(0.3, max(res["verdict_flip_rate"].max(), res["framework_shift_rate"].max()) * 1.3))
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig5_thinking_effect.pdf", bbox_inches="tight")
        plt.close()
    return {"summary": res}


# ─── LaTeX tables ───────────────────────────────────────────────────────────

def write_latex_tables(m1, m2, m3, m4, m7):
    # Overall summary table: each row is a config; columns are key metrics
    rows = []
    for cfg in CONFIG_ORDER:
        fam, mode = cfg.split("_")
        m3_rate = float(m3[m3["config_key"] == cfg]["rate"].iloc[0]) if cfg in m3["config_key"].values else float("nan")
        m4_rate = float(m4[m4["config_key"] == cfg]["bias_coefficient"].iloc[0]) if cfg in m4["config_key"].values else float("nan")
        rows.append({"Configuration": f"{MODEL_PRETTY[fam]} ({MODE_PRETTY[mode]})",
                     "Paraphrase consistency": f"{m3_rate:.2f}",
                     "Demographic bias": f"{m4_rate:.2f}"})
    df = pd.DataFrame(rows)
    tex = df.to_latex(index=False, escape=False,
                      caption="Per-configuration consistency and bias metrics. "
                              "Higher paraphrase consistency = more semantically robust. "
                              "Lower demographic bias = more demographically blind.",
                      label="tab:summary")
    (TBL_DIR / "tab_summary.tex").write_text(tex)

    # Krippendorff alpha summary with bootstrap CIs
    if m2:
        alpha_lines = ["\\begin{tabular}{lrrr}", "\\toprule",
                       "Mode & Krippendorff's $\\alpha$ & 95\\% CI & Scenarios \\\\",
                       "\\midrule"]
        for mode in ["instant", "light"]:
            if mode in m2:
                a = m2[mode]["alpha"]
                lo, hi = m2[mode].get("alpha_ci", (float("nan"), float("nan")))
                alpha_lines.append(f"{MODE_PRETTY[mode]} & "
                                   f"{a:.3f} & "
                                   f"[{lo:.3f}, {hi:.3f}] & "
                                   f"{m2[mode]['n_scenarios']} \\\\")
        alpha_lines += ["\\bottomrule", "\\end{tabular}"]
        (TBL_DIR / "tab_alpha.tex").write_text("\n".join(alpha_lines))

    # Thinking effect table with bootstrap CIs
    if "summary" in m7 and not m7["summary"].empty:
        lines = ["\\begin{tabular}{lcccr}", "\\toprule",
                 "Model family & Verdict-flip [95\\% CI] & Framework-shift [95\\% CI] & $n$ \\\\",
                 "\\midrule"]
        for _, r in m7["summary"].iterrows():
            v_str = f"{r['verdict_flip_rate']:.2f} [{r['verdict_ci_lo']:.2f}, {r['verdict_ci_hi']:.2f}]"
            f_str = f"{r['framework_shift_rate']:.2f} [{r['fw_ci_lo']:.2f}, {r['fw_ci_hi']:.2f}]"
            lines.append(f"{MODEL_PRETTY[r['family']]} & {v_str} & {f_str} & "
                         f"{int(r['n_scenarios'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TBL_DIR / "tab_thinking_effect.tex").write_text("\n".join(lines))


# ─── Console summary ────────────────────────────────────────────────────────

def print_summary(df, m1, m2, m3, m4, m5, m6, m7):
    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    n_total = len(df); n_ok = int(df["ok"].sum())
    print(f"  Records: {n_total} loaded, {n_ok} successful ({100*n_ok/n_total:.1f}%)")
    print(f"\n  Per-config record counts (successful only):")
    for cfg in CONFIG_ORDER:
        n = int(((df["config_key"] == cfg) & df["ok"]).sum())
        print(f"    {cfg:<20} {n:>4}")

    print("\n  M1 — Framework distribution (top 2 per config):")
    for cfg, row in m1.iterrows():
        top = row.sort_values(ascending=False).head(2)
        print(f"    {cfg:<20} {dict(top.round(2))}")

    print("\n  M2 — Cross-model agreement:")
    for mode in ["instant", "light"]:
        if mode in m2:
            print(f"    {mode:<10} α={m2[mode]['alpha']:.3f}  ({m2[mode]['n_scenarios']} scenarios)")

    if not m3.empty:
        print("\n  M3 — Paraphrase consistency:")
        for _, r in m3.iterrows():
            print(f"    {r['config_key']:<20} {r['rate']:.2f}  ({r['consistent']}/{r['total']})")

    if not m4.empty:
        print("\n  M4 — Demographic bias coefficient:")
        for _, r in m4.iterrows():
            print(f"    {r['config_key']:<20} {r['bias_coefficient']:.2f}")

    if "summary" in m7 and not m7["summary"].empty:
        print("\n  M7 — Effect of thinking:")
        for _, r in m7["summary"].iterrows():
            print(f"    {MODEL_PRETTY[r['family']]:<20} verdict-flip={r['verdict_flip_rate']:.2f}  "
                  f"framework-shift={r['framework_shift_rate']:.2f}")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_results()
    meta = load_scenario_metadata()
    m1 = m1_framework_distribution(df)
    m2 = m2_cross_model_agreement(df)
    m2b = m2b_easy_vs_hard_alpha(df, m2)
    m3 = m3_paraphrase_consistency(df, meta)
    m4 = m4_demographic_bias(df, meta)
    m5 = m5_hard_case_entropy(df)
    m6 = m6_confidence(df)
    m_tc = m_trace_concordance(df)
    m_rt = m_reasoning_tokens(df)
    m7 = m7_thinking_effect(df)
    write_latex_tables(m1, m2, m3, m4, m7)
    print_summary(df, m1, m2, m3, m4, m5, m6, m7)
    if m2b:
        print("\n  H4 — Easy vs hard α:")
        for diff in ("easy", "hard"):
            n = m2b[diff]["n"]
            for mode in ("instant", "light"):
                if mode in m2b[diff]:
                    a, lo, hi = m2b[diff][mode]
                    print(f"    {diff:<6} {mode:<8} n={n:>3}  α={a:.3f} [{lo:.3f}, {hi:.3f}]")
    if not m_tc.empty:
        print("\n  Trace concordance (self-label vs trace-dominant framework):")
        for _, r in m_tc.iterrows():
            fam_p = MODEL_PRETTY.get(r["family"], r["family"])
            if pd.isna(r["concordance"]):
                print(f"    {fam_p:<22} no traces with detectable framework cues")
            else:
                print(f"    {fam_p:<22} {r['concordance']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  "
                      f"(n={int(r['n_with_cues'])}/{int(r['n_total'])})")


if __name__ == "__main__":
    main()
