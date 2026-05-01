# How Does Thinking Change the Morality of LLMs?

A controlled study of reasoning effects on moral judgment in five frontier
language models, using single-checkpoint instant-vs-thinking comparisons.

This repository contains the benchmark, experiment runner, analysis pipeline,
raw model outputs, and the LaTeX source for the accompanying paper.

## TL;DR

We query five frontier reasoning-capable LLMs on 100 ethical scenarios in
both **instant** and **lightweight thinking** modes. All five families use a
*single-checkpoint* comparison (same weights, only the reasoning toggle
changes), which removes capability confounds from the comparison.

| Lab | Model | Instant config | Thinking config |
|---|---|---|---|
| Anthropic | `claude-sonnet-4-6` | omit thinking param | `thinking={enabled, budget_tokens=1024}` |
| OpenAI | `gpt-5.5` | `reasoning.effort=none` | `reasoning.effort=medium` |
| Google DeepMind | `gemini-3-flash-preview` | `thinking_level=minimal` | `thinking_level=low` |
| DeepSeek | `deepseek-ai/DeepSeek-V3.1` | `reasoning.enabled=False` | `reasoning.enabled=True` |
| Alibaba | `Qwen/Qwen3.5-397B-A17B` | `reasoning.enabled=False` | `reasoning.enabled=True` |

### Headline results

- Aggregate cross-model agreement on binary verdicts is high in both modes (Krippendorff's α = 0.78 / 0.79) but masks an easy/hard split.
- On 21 contested "hard" scenarios, instant-mode agreement is at chance level (α = 0.08); reasoning improves it modestly (α = 0.23) — directional only.
- Reasoning reduces demographic-judgment inconsistency in 3 of 5 models; increases it for none.
- Self-labeled ethical framework matches the reasoning trace ≥90% for DeepSeek/Gemini/Qwen3.5, but only 24% for Claude and 58% for GPT-5.5.
- "Lightweight thinking" is *not* comparable across providers: per-call reasoning-token spend ranges from 33 (Claude) to 2,639 (Qwen3.5).

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (copy .env.example to .env and edit)
cp .env.example .env

# 3. Run the smoketest (~$0.50, 3 min) to verify your setup
python experiments/smoketest.py

# 4. Run the full experiment (~$30, 40 min wall-clock at 10-way parallel)
python experiments/run_experiments.py --runs 3

# 5. Generate figures, tables, and CSVs
python experiments/analyze_results.py
```

The runner is **resumable**: it writes each result to disk immediately and on
relaunch skips any (config × scenario × run) tuple that already has a
successful record. You can kill and restart at any time without losing or
re-paying for prior calls.

## Repository layout

```
morality_of_llms/
├── experiments/
│   ├── scenarios/                  # 100 scenarios across 5 categories (JSON)
│   │   ├── trolley_problems.json
│   │   ├── moral_foundations.json
│   │   ├── paraphrase_consistency.json
│   │   ├── demographic_sensitivity.json
│   │   └── contemporary_dilemmas.json
│   ├── models/                     # Provider client wrappers
│   │   ├── base.py                 # Strict-JSON parsing, common ModelResponse
│   │   ├── anthropic_client.py     # Adaptive thinking
│   │   ├── openai_client.py        # Responses API + reasoning effort
│   │   ├── gemini_client.py        # thinking_level (3.x canonical)
│   │   └── together_client.py      # reasoning.enabled (canonical)
│   ├── run_experiments.py          # Resumable runner with smart retries
│   ├── analyze_results.py          # Computes 7 metrics + emits figures/tables
│   ├── smoketest.py                # End-to-end pipeline test (1 scenario × 10 configs)
│   └── peek.py                     # Inspect full structured responses for a few scenarios
├── results/
│   ├── raw/                        # Per-config JSONL output (append-only)
│   ├── processed/                  # Computed metrics as CSV
│   └── smoketest/                  # Smoketest results
├── paper/
│   ├── main.tex                    # Manuscript source
│   ├── references.bib              # Bibliography
│   ├── figures/                    # Auto-generated PDF figures
│   └── tables/                     # Auto-generated LaTeX tables
├── requirements.txt
├── .env.example
└── LICENSE                         # MIT
```

## Reproducing the experiment

### Required API keys

You need accounts with:

| Provider | Where to get keys | Estimated spend (full run) |
|---|---|---|
| Anthropic | <https://console.anthropic.com/> | ~$10 |
| OpenAI | <https://platform.openai.com/> | ~$6 |
| Google AI Studio | <https://aistudio.google.com/> (paid tier required for our quota) | ~$2 |
| Together AI | <https://api.together.xyz/> | ~$5 |

Total: ~$25 for one full N=3 run; we recommend ~$55 in account balance to allow for retries and longer thinking traces.

### Running

```bash
# All configs, 3 samples per scenario (default)
python experiments/run_experiments.py

# Subset of categories (tp = trolley, mf = moral foundations, pc = paraphrase,
# ds = demographic, cd = contemporary)
python experiments/run_experiments.py --categories tp mf

# Subset of model configurations
python experiments/run_experiments.py --configs claude_instant claude_light gpt55_light

# Custom number of stochastic runs (lower N = cheaper, less statistical power)
python experiments/run_experiments.py --runs 1
```

The runner writes one JSONL file per configuration to `results/raw/`. Each
line is one (config × scenario × run) record with the parsed judgment,
reasoning text, latency, and any error. Resume by simply re-running the
command; already-completed records are skipped.

### Analyzing

```bash
python experiments/analyze_results.py
```

Outputs:
- `results/processed/m{1..7}_*.csv` — machine-readable metric summaries
- `paper/figures/fig{1..5}_*.pdf` — figures embedded in the manuscript
- `paper/tables/tab_*.tex` — auto-populated LaTeX tables
- Console summary of all metrics

### Building the paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Methodology

The full methodology, hypotheses (H1–H5), prompt template, and metric
definitions are in `paper/main.tex`. The 100 scenarios are documented inline
in `experiments/scenarios/*.json` with category, expected framework (where
applicable), and pedagogical notes.

## Limitations

See the Limitations section of the paper. Briefly: (1) self-labeled ethical
frameworks are not externally validated; (2) 100 scenarios is small relative
to ETHICS-scale benchmarks; (3) N=3 is moderate statistical power; (4) we use
`gemini-3-flash-preview` rather than the flagship Gemini 3.1 Pro Preview
because the latter does not allow `thinking_level=minimal`, preventing a
single-checkpoint instant comparison; (5) all scenarios are in English; (6)
"lightweight thinking" is not a comparable construct across providers
(reasoning-token spend ranges from 33 to 2,639 per call).

## Citation

If you use this benchmark or pipeline, please cite:

```bibtex
@article{madur2026thinking,
  title  = {How Does Thinking Change the Morality of LLMs?
            A Controlled Instant-vs-Thinking Comparison Across
            Five Frontier Reasoning-Trained Language Models},
  author = {Madur, Sourabh},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
