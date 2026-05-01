#!/usr/bin/env python3
"""
Lightweight LaTeX manuscript validator.

Without a real pdflatex install, verifies:
  - all \\input{} files exist
  - all \\includegraphics{} files exist
  - all \\cite{} keys are present in references.bib
  - all \\ref{} have a corresponding \\label{}
  - \\begin/\\end environments are balanced
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MAIN = ROOT / "main.tex"
BIB  = ROOT / "references.bib"

def expand_inputs(text: str, base: Path) -> str:
    """Inline \\input{} files so labels/refs resolve correctly."""
    def repl(m):
        name = m.group(1)
        f = base / (name + ("" if name.endswith(".tex") else ".tex"))
        return f.read_text() if f.exists() else m.group(0)
    return re.sub(r"\\input\{([^}]+)\}", repl, text)

src = expand_inputs(MAIN.read_text(), ROOT)
bib = BIB.read_text() if BIB.exists() else ""

errors = []
warnings = []

# 1. \input{...}
for m in re.finditer(r"\\input\{([^}]+)\}", src):
    f = ROOT / (m.group(1) + (".tex" if not m.group(1).endswith(".tex") else ""))
    if not f.exists():
        errors.append(f"Missing \\input file: {f}")

# 2. \includegraphics{...}
for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
    name = m.group(1)
    candidates = [ROOT / name, ROOT / (name + ".pdf"), ROOT / (name + ".png")]
    if not any(c.exists() for c in candidates):
        errors.append(f"Missing graphics file: {name} (tried .pdf/.png)")

# 3. \cite{...}
all_cites = set()
for m in re.finditer(r"\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]+)\}", src):
    keys = [k.strip() for k in m.group(1).split(",")]
    all_cites.update(keys)

bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
for c in all_cites:
    if c not in bib_keys:
        errors.append(f"\\cite key not in .bib: {c}")
unused = bib_keys - all_cites
if unused:
    warnings.append(f"Unused .bib entries: {sorted(unused)}")

# 4. \ref{} ↔ \label{}
labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
for r in refs:
    if r not in labels:
        errors.append(f"\\ref to undefined label: {r}")

# 5. \begin/\end balance
begins = re.findall(r"\\begin\{([^}]+)\}", src)
ends = re.findall(r"\\end\{([^}]+)\}", src)
from collections import Counter
b, e = Counter(begins), Counter(ends)
all_envs = set(b.keys()) | set(e.keys())
for env in all_envs:
    if b[env] != e[env]:
        errors.append(f"Unbalanced environment '{env}': {b[env]} \\begin vs {e[env]} \\end")

# Stats
n_words = len(re.findall(r"\b\w+\b", src))
n_figs = len(re.findall(r"\\begin\{figure\}", src))
n_tables = len(re.findall(r"\\begin\{table\}|\\input\{tables/", src))

print("=" * 60)
print(f"  LaTeX validation — {MAIN}")
print("=" * 60)
print(f"  Source: {len(src):,} chars, {n_words:,} words")
print(f"  Figures: {n_figs}, Tables: {n_tables}")
print(f"  Citations: {len(all_cites)}, .bib entries: {len(bib_keys)}")
print(f"  \\labels: {len(labels)}, \\refs: {len(refs)}")
print()
if errors:
    print(f"  ❌ {len(errors)} ERRORS:")
    for e in errors: print(f"     - {e}")
else:
    print("  ✅ No errors found.")
if warnings:
    print(f"\n  ⚠️  {len(warnings)} WARNINGS:")
    for w in warnings: print(f"     - {w}")
print()
sys.exit(1 if errors else 0)
