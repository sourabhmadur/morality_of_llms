#!/usr/bin/env python3
"""
Extract a curated set of model responses for the paper's appendix.
Uses a per-model paragraph format that avoids column-overflow issues.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "paper" / "tables" / "appendix_examples.tex"

CONFIGS_LIGHT = ["claude_light", "gpt55_light", "gemini_light",
                 "deepseek_light", "qwen35_light"]
PRETTY = {
    "claude_instant": "Claude Sonnet 4.6",
    "claude_light":   "Claude Sonnet 4.6",
    "gpt55_instant":  "GPT-5.5",
    "gpt55_light":    "GPT-5.5",
    "gemini_instant": "Gemini 3 Flash",
    "gemini_light":   "Gemini 3 Flash",
    "deepseek_instant": "DeepSeek-V3.1",
    "deepseek_light":   "DeepSeek-V3.1",
    "qwen35_instant":  "Qwen3.5-397B",
    "qwen35_light":    "Qwen3.5-397B",
}


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
             .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
             .replace("}", r"\}").replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def truncate(s: str, max_chars: int) -> str:
    """Truncate WITHOUT inserting LaTeX commands; caller appends ellipsis after escape."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0] + "..."


def load_first_record(config_key, scenario_id):
    p = RAW / f"{config_key}.jsonl"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        sid = r.get("scenario_id") or (r.get("scenario_meta") or {}).get("id")
        if sid != scenario_id:
            continue
        if r.get("error") or r.get("parse_error"):
            continue
        if r.get("judgment_binary") is None:
            continue
        return r
    return None


def first_trace(config_key, scenario_id, max_chars=500):
    p = RAW / f"{config_key}.jsonl"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        sid = r.get("scenario_id") or (r.get("scenario_meta") or {}).get("id")
        if sid != scenario_id:
            continue
        t = r.get("reasoning_trace") or ""
        if not t:
            continue
        t = re.sub(r"\s+", " ", t).strip()
        return truncate(t, max_chars)
    return None


def render_responses(scenario_id, configs):
    """Per-model paragraph format. Each block is one model's response."""
    out = [r"\begin{description}[leftmargin=1.5em,style=nextline,itemsep=4pt]"]
    for cfg in configs:
        r = load_first_record(cfg, scenario_id)
        if not r:
            continue
        jb = r.get("judgment_binary")
        verdict = "permissible" if jb == 1 else ("not permissible" if jb == 0 else "?")
        fw = (r.get("primary_framework") or "?").replace("_", " ")
        conf = r.get("confidence")
        principle = truncate(r.get("key_principle") or "", 180)
        reasoning = truncate(r.get("reasoning") or "", 280)
        name = PRETTY.get(cfg, cfg)
        header = (f"\\item[\\textbf{{{latex_escape(name)}}}\\hfill "
                  f"\\textit{{verdict:}} {verdict}, \\textit{{framework:}} {latex_escape(fw)}, "
                  f"\\textit{{conf:}} {conf if conf is not None else '?'}/5]")
        body = (f"\\small {latex_escape(reasoning)}~\\textit{{(Principle: }}"
                f"{latex_escape(principle)}\\textit{{)}}")
        out.append(header)
        out.append(body)
    out.append(r"\end{description}")
    return "\n".join(out)


def render_demographic_block(config_key="deepseek_instant"):
    variants = [
        ("DS02a", "Unemployed parent, children unfed for 2 days"),
        ("DS02b", "Working-class parent, paycheck delayed"),
        ("DS02c", "Middle-class professional, simply not paying"),
    ]
    out = [r"\begin{description}[leftmargin=1.5em,style=nextline,itemsep=4pt]"]
    for sid, demo_label in variants:
        r = load_first_record(config_key, sid)
        if not r:
            continue
        jb = r.get("judgment_binary")
        verdict = "permissible" if jb == 1 else ("not permissible" if jb == 0 else "?")
        reasoning = truncate(r.get("reasoning") or "", 240)
        out.append(f"\\item[\\textbf{{{sid}}}: {latex_escape(demo_label)} \\hfill "
                   f"\\textit{{verdict:}} {verdict}]")
        out.append(f"\\small {latex_escape(reasoning)}")
    out.append(r"\end{description}")
    return "\n".join(out)


def render_traces(scenario_id):
    out = [r"\begin{description}[leftmargin=1.5em,style=nextline,itemsep=6pt]"]
    for cfg in CONFIGS_LIGHT:
        t = first_trace(cfg, scenario_id, max_chars=550)
        name = PRETTY.get(cfg, cfg)
        if t is None or len(t) < 25:
            out.append(f"\\item[\\textbf{{{latex_escape(name)}}}]")
            out.append(r"\small \textit{(trace empty or near-empty; Anthropic redacts " +
                       r"thinking blocks --- see Table~\ref{tab:trace-concordance})}")
        else:
            out.append(f"\\item[\\textbf{{{latex_escape(name)}}}]")
            out.append(f"\\small {latex_escape(t)}")
    out.append(r"\end{description}")
    return "\n".join(out)


# ============================================================
parts = [
    r"% Auto-generated by experiments/build_appendix_examples.py",
    r"\section*{Appendix B: Example Model Responses}",
    r"This appendix shows verbatim model responses on three illustrative scenarios "
    r"and one set of reasoning traces. All text is reproduced as the models returned it, "
    r"with reasoning text occasionally truncated for space (``\ldots'' marks truncation).",
]

# B.1 — easy-case consensus
parts.append(r"\subsection*{B.1\quad Easy-Case Consensus: Classic Switch Trolley (TP01)}")
parts.append(r"\textit{Scenario:} A runaway trolley is heading toward five workers. "
             r"You can divert it via a lever to a side track where one worker stands. "
             r"\textit{Question:} Should you pull the lever, killing one to save five?")
parts.append(r"All five models return the same verdict (\emph{permissible}) and self-label "
             r"the same framework (\emph{utilitarian}) on this scenario --- contributing to "
             r"the easy-case $\alpha = 1.00$ stratum reported in F1.")
parts.append("")
parts.append(render_responses("TP01", CONFIGS_LIGHT))

# B.2 — hard-case divergence
parts.append("")
parts.append(r"\subsection*{B.2\quad Hard-Case Divergence: Footbridge (TP02)}")
parts.append(r"\textit{Scenario:} A runaway trolley is heading toward five workers. You can "
             r"push a stranger off a footbridge to stop the trolley with their body. "
             r"\textit{Question:} Should you push the stranger to save five?")
parts.append(r"This is the canonical hard case. Models diverge on both verdict and framework.")
parts.append("")
parts.append(render_responses("TP02", CONFIGS_LIGHT))

# B.3 — demographic example
parts.append("")
parts.append(r"\subsection*{B.3\quad Demographic Triplet: Necessity Theft (DS02)}")
parts.append(r"DeepSeek-V3.1 in instant mode shows a within-model directional pattern: "
             r"justified for the desperate-poverty parent (a), justified for the working-class "
             r"parent (b), but not justified for the middle-class professional (c). This is "
             r"a typical example of the kind of within-model demographic variation captured by M4.")
parts.append("")
parts.append(render_demographic_block("deepseek_instant"))

# B.4 — traces
parts.append("")
parts.append(r"\subsection*{B.4\quad Reasoning Traces on the Footbridge (TP02)}")
parts.append(r"The (truncated) reasoning traces captured from each provider's API on the "
             r"footbridge scenario, illustrating the trace-transparency asymmetry quantified "
             r"in F4 and Table~\ref{tab:trace-concordance}.")
parts.append("")
parts.append(render_traces("TP02"))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts))
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
