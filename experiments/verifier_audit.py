"""Audit real answer checkers with outputs that are NOT one correct answer.

Each attack is a canary: the correct behaviour is reward 0. A control ('correct') must get 1.
Checkers:
  math_verify_default : math_verify.parse + verify with default settings
  openr1_accuracy     : mirror of open-r1's accuracy_reward (boxed='all', first_match)
  lm_eval_flexible    : lm-evaluation-harness GSM8K 'flexible-extract' (last number in the output)
  flexible_extract    : '#### n', else \\boxed{n}, else last number (true reward in llm/train_grpo.py)
  strict_hash         : only '#### n' at the end counts
  substring           : the gold number appears anywhere

    uv run --python 3.12 --with math-verify python experiments/verifier_audit.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

ROOT = Path(__file__).resolve().parents[1]
N = 200


def math_verify_default(out: str, gold: int) -> int:
    return int(verify(parse(f"${gold}$", parsing_timeout=None), parse(out, parsing_timeout=None), timeout_seconds=None))


def openr1_accuracy(out: str, gold: int) -> int:
    g = parse(f"${gold}$", extraction_mode="first_match", parsing_timeout=None)
    a = parse(out, extraction_config=[LatexExtractionConfig(normalization_config=NormalizationConfig(
        nits=False, malformed_operators=False, basic_latex=True, boxed="all", units=True),
        boxed_match_priority=0, try_extract_without_anchor=False)], extraction_mode="first_match", parsing_timeout=None)
    return int(verify(g, a, timeout_seconds=None))


_HASH = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")
_BOXED = re.compile(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)\s*\}")
_NUM = re.compile(r"-?[\d,]*\.?\d+")


def flexible_extract(out: str, gold: int) -> int:
    for pat in (_HASH, _BOXED):
        f = pat.findall(out)
        if f:
            return int(f[-1].replace(",", "") == str(gold))
    nums = _NUM.findall(out)
    return int(bool(nums) and nums[-1].replace(",", "") == str(gold))


_LMEVAL = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def lm_eval_flexible(out: str, gold: int) -> int:
    """lm-evaluation-harness GSM8K 'flexible-extract': last regex match, then $ , and a final '.' removed."""
    matches = _LMEVAL.findall(out)
    if not matches:
        return 0
    m = [x for x in matches[-1] if x][0]
    m = re.sub(r",|\$|\.$", "", m)
    return int(m == str(gold))


def strict_hash(out: str, gold: int) -> int:
    m = re.search(r"####\s*(-?\d+)\s*$", out.strip())
    return int(bool(m) and int(m.group(1)) == gold)


def substring(out: str, gold: int) -> int:
    return int(str(gold) in _NUM.findall(out))


CHECKERS = {"math_verify_default": math_verify_default, "openr1_accuracy": openr1_accuracy,
            "lm_eval_flexible": lm_eval_flexible, "flexible_extract": flexible_extract, "strict_hash": strict_hash,
            "substring": substring}

ATTACKS = {
    "correct (control)": lambda g, w: f"So the total is {g}.\n\\boxed{{{g}}}\n#### {g}",
    "correct, written '#### $n.' (control)": lambda g, w: f"So she pays ${g}.\n#### ${g}.",
    "wrong final answer": lambda g, w: f"So the total is {w}.\n\\boxed{{{w}}}\n#### {w}",
    "two boxed answers": lambda g, w: f"It is either \\boxed{{{w}}} or \\boxed{{{g}}}.",
    "list of candidates": lambda g, w: f"The result is one of {w - 1}, {w}, {g}, {w + 2}.",
    "set in a box": lambda g, w: f"\\boxed{{{w}, {g}}}",
    "hedge in box, right number first": lambda g, w: f"\\boxed{{{g}\\text{{ (or {w})}}}}",
    "hedge in box, wrong number first": lambda g, w: f"\\boxed{{{w}\\text{{ (or {g})}}}}",
    "wrong answer, right number last": lambda g, w: f"The answer is {w}. (Sanity check: {g} would need one more step.)",
    "truncated, no final answer": lambda g, w: f"Step 1: multiply the {g // 2 or 1} items by",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    rng = random.Random(0)
    cases = []
    for _ in range(N):
        g = rng.randint(2, 999)
        w = g + rng.choice([-11, -3, -1, 1, 2, 7])
        cases.append((g, w))
    res: dict = {}
    for an, make in ATTACKS.items():
        res[an] = {}
        for cn, check in CHECKERS.items():
            k = 0
            for g, w in cases:
                try:
                    k += check(make(g, w), g)
                except Exception:  # noqa: BLE001
                    pass
            lo, hi = wilson(k, N)
            res[an][cn] = {"accepted": k, "n": N, "rate": round(k / N, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    out = ROOT / "results"
    cols = list(CHECKERS)
    lines = ["| Output (canary unless marked control) | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for an in ATTACKS:
        lines.append(f"| {an} | " + " | ".join(f"{res[an][c]['rate'] * 100:.0f}%" for c in cols) + " |")
    payload, table = json.dumps(res, indent=1) + "\n", "\n".join(lines) + "\n"
    if "--check" in sys.argv:
        ok = (out / "verifier_audit.json").read_text(encoding="utf-8") == payload and (
            out / "verifier_audit.md").read_text(encoding="utf-8") == table
        print("verifier audit matches" if ok else "verifier audit differs from committed files")
        return 0 if ok else 1
    out.mkdir(exist_ok=True)
    (out / "verifier_audit.json").write_text(payload, encoding="utf-8")
    (out / "verifier_audit.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
