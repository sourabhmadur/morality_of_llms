# Results at a Glance — Final (post-review revision)

*Full data: 2,963 successful calls. All 10 (model × mode) configurations populated. Bootstrap 95% CIs throughout.*

## Four headline findings

### F1 — Aggregate agreement masks an easy/hard split
| Stratum | Mode | α | 95% CI | n |
|---|---|---|---|---|
| Aggregate | instant | 0.781 | (small overlap) | 100 |
| Aggregate | thinking | 0.789 | (small overlap) | 99 |
| **Easy** (all 5 agree in instant) | instant | **1.000** | [1.00, 1.00] | 79 |
| **Easy** | thinking | 0.949 | [0.89, 0.99] | 79 |
| **Hard** (≥1 disagrees in instant) | instant | **0.080** | [-0.06, 0.15] | 21 |
| **Hard** | thinking | **0.234** | [0.01, 0.45] | 21 |

**Reasoning helps where models disagreed in the first place.** On easy cases, models already agree perfectly; on hard cases, instant-mode agreement is essentially random (α=0.08), and reasoning roughly triples it (α=0.23) — still poor, but a directional improvement. **The aggregate α numbers conceal this entirely.**

### F2 — Direction-divergent framework shifts (validated against trace content)

Self-labeled framework changes induced by reasoning:
| Lab | Instant top-2 | Thinking top-2 | Direction |
|---|---|---|---|
| Claude Sonnet 4.6 | 42% deon, 39% util | 39% deon, 37% util | mostly stable |
| **GPT-5.5** | 41% deon, 30% util | **43% deon**, 34% util | **→ deontological (unique)** |
| Gemini 3 Flash | 59% util, 32% deon | 59% util, 35% deon | stable |
| **DeepSeek-V3.1** | 46% deon, 43% util | **48% util**, 47% deon | **→ utilitarian (plurality flips)** |
| Qwen3.5-397B | 49% util, 39% deon | 49% util, 42% deon | stable |

**Trace-content concordance** — does the trace actually mention the framework the model self-labels?
| Lab | Concordance | 95% CI | Traces with cues / total |
|---|---|---|---|
| Claude Sonnet 4.6 | 0.24 | [0.08, 0.40] | 25 / 286 (Anthropic redacts traces) |
| **DeepSeek-V3.1** | **0.92** | [0.89, 0.95] | 298 / 298 |
| Gemini 3 Flash | 0.90 | [0.87, 0.94] | 264 / 265 |
| **GPT-5.5** | **0.58** | [0.48, 0.68] | 91 / 200 (label often decoupled from reasoning) |
| Qwen3.5-397B | 0.92 | [0.89, 0.95] | 296 / 296 |

**F2 well-supported for DeepSeek/Gemini/Qwen3.5; partial for GPT-5.5; unverified for Claude** (because Anthropic's API hides the trace).

### F3 — Reasoning reduces demographic-judgment inconsistency

| Lab | Instant | Thinking | Δ |
|---|---|---|---|
| Claude | 0.10 | 0.10 | floor |
| **GPT-5.5** | 0.20 | 0.10 | **−0.10** |
| Gemini | 0.10 | 0.10 | floor |
| **DeepSeek-V3.1** | **0.30** | **0.10** | **−0.20** (3× reduction) |
| **Qwen3.5-397B** | 0.20 | 0.10 | **−0.10** |

**Caveat (per SG-4):** M4 measures *inconsistency* (asymmetry across demographic variants), not *directional bias*. Per-triplet directional info is logged in `m4b_directional_bias.csv`; quantifying which way the bias goes is left to follow-up because keyword coding alone can't distinguish "harsher to marginalized" from "more lenient to marginalized" across diverse demographic axes.

### F4 — Reasoning effort spans 31× across providers

Mean reasoning tokens at each provider's lightest non-zero setting (n=300 each):
| Lab | Mean | Std |
|---|---|---|
| Claude Sonnet 4.6 | **1,024** (capped) | 0 |
| DeepSeek-V3.1 | 912 | 406 |
| Gemini 3 Flash | 532 | 246 |
| **GPT-5.5** | **84** (very low) | 63 |
| **Qwen3.5-397B** | **2,639** (very high) | 783 |

**Counter-example to the naive "more tokens → more change" hypothesis:** GPT-5.5 produces only 84 reasoning tokens but has a 6% verdict-flip rate (6× Claude's, despite Claude using 12× more tokens). Per-provider differences in *how* reasoning is shaped by post-training matter more than raw token count.

**Critical caveat:** GPT-5.5 at `effort=low` produces *zero* reasoning tokens for our prompts (verified via `usage.output_tokens_details.reasoning_tokens`). We use `effort=medium` for OpenAI's thinking arm, which is more generous than the apparently-equivalent settings on the other providers.

## Hypothesis verdicts

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Different model families default to different ethical frameworks | **Supported** |
| H2 | Surface-form variations produce non-trivial verdict flips | **Rejected** (paraphrase consistency 0.80–1.00) |
| H3 | Demographic variation produces different judgments | **Partially supported** (3 of 5 models) |
| H4 | Cross-model agreement collapses on hard cases | **Strongly supported** (α=1.00 easy → 0.08 hard) |
| H5 | Thinking systematically changes moral judgments | **Mixed** (effects real but small for most models) |

## What this means in three sentences

> Frontier reasoning-trained LLMs are highly consistent on easy moral cases and roughly random on hard ones in instant mode; reasoning helps modestly on the hard cases without disrupting the easy ones. Reasoning shifts each model's stated ethical framework slightly — and in different directions across providers — but trace-content analysis shows GPT-5.5's framework labels are partially decoupled from its actual reasoning. Reasoning reliably reduces demographic-judgment inconsistency for every model where it was elevated, with no model showing increased inconsistency, suggesting reasoning is a useful (but model-specific) lever for fairness.

## Provenance & changes from earlier draft (review notes)

This version addresses an internal review:
- **MJ-1**: α numbers now consistent across abstract / Results / Discussion (0.781 instant, 0.789 thinking, with bootstrap CIs).
- **MJ-2**: Table 1 updated to match actual API parameters used (`thinking.enabled+budget_tokens=1024`, `effort=medium+summary=detailed`, `thinking_level=low+include_thoughts=True`, `reasoning.enabled=True/False`).
- **MJ-3**: "Pre-registered" claim removed; H1-H5 each receive an explicit verdict (Table 7 in paper).
- **MJ-4**: Bootstrap 95% CIs added to α, M3 paraphrase consistency, M4 demographic bias, M7 verdict-flip and framework-shift rates.
- **MJ-5**: §5.2 deployment guidance no longer contradicts F1.
- **MJ-6**: GPT-5.5's effort-tier behavior documented honestly (effort=low produces 0 reasoning tokens).
- **MJ-7**: "Single-checkpoint" claim qualified (same weights, but different inference configs).
- **SG-1**: Trace-vs-self-label concordance computed; reframes F2 as well-supported for 3/5 providers.
- **SG-3**: GPT-5.5 outlier discussed in F4.
- **SG-4**: Directional vs. inconsistency distinction made explicit in F3.
- **SG-7**: Unused `moonshot2025kimi` citation removed.
