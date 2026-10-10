"""Regenerate results/summary.json and results/table.md from fixed seeds.

    uv run python experiments/run_demo.py          # write results
    uv run python experiments/run_demo.py --check  # fail if committed results differ
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reward_audit.monitor import zero_defect_sample_size  # noqa: E402
from reward_audit.train import run  # noqa: E402

SEEDS = list(range(40))
VERIFIER_ORDER = ("exact", "exact_noisy", "substring", "judge_like")


def _last(curve: list[float], k: int = 10) -> float:
    return statistics.mean(curve[-k:])


def _med(xs: list) -> float | None:
    return statistics.median(xs) if xs else None


def summarize() -> dict:
    out: dict = {"seeds": len(SEEDS), "steps": 150, "batch": 32, "audit_per_step": 2,
                 "eps": 0.02, "alpha": 0.05, "verifiers": {}}
    for name in VERIFIER_ORDER:
        runs = [run(name, s) for s in SEEDS]
        flagged = [r for r in runs if r.flagged_at is not None]
        taken = [r for r in runs if r.takeover_step is not None]
        out["verifiers"][name] = {
            "final_reward": round(statistics.mean(_last(r.reward_curve) for r in runs), 3),
            "final_true_accuracy": round(statistics.mean(_last(r.true_curve) for r in runs), 3),
            "final_false_reward_rate": round(statistics.mean(_last(r.false_reward_curve) for r in runs), 3),
            "final_exploit_share": round(statistics.mean(r.exploit_curve[-1] for r in runs), 3),
            "runs_taken_over": len(taken),
            "median_takeover_step": _med([r.takeover_step for r in taken]),
            "runs_flagged": len(flagged),
            "median_flag_step": _med([r.flagged_at for r in flagged]),
            "median_audit_labels_to_flag": _med([r.audited for r in flagged]),
            "flagged_before_takeover": sum(
                1 for r in taken if r.flagged_at is not None and r.flagged_at < r.takeover_step
            ),
        }
    out["zero_defect_checks_for_5pct_at_95pct"] = zero_defect_sample_size(0.05, 0.95)
    return out


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:g}" if isinstance(x, (int, float)) else str(x))


def table(s: dict) -> str:
    n = s["seeds"]
    lines = [
        "| Reward source | Reward | True accuracy | False-reward rate | Exploit share | Taken over | Median takeover step | Flagged | Median flag step | Median audit labels |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, v in s["verifiers"].items():
        lines.append(
            f"| {name} | {v['final_reward']:.2f} | {v['final_true_accuracy']:.2f} | {v['final_false_reward_rate']:.3f} | "
            f"{v['final_exploit_share']:.2f} | {v['runs_taken_over']}/{n} | {_fmt(v['median_takeover_step'])} | "
            f"{v['runs_flagged']}/{n} | {_fmt(v['median_flag_step'])} | {_fmt(v['median_audit_labels_to_flag'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    s = summarize()
    payload = json.dumps(s, indent=2, sort_keys=True) + "\n"
    tbl = table(s)
    res = ROOT / "results"
    if "--check" in sys.argv:
        ok = (res / "summary.json").read_text(encoding="utf-8") == payload and (
            res / "table.md"
        ).read_text(encoding="utf-8") == tbl
        print("results match" if ok else "results differ from committed files")
        return 0 if ok else 1
    res.mkdir(exist_ok=True)
    (res / "summary.json").write_text(payload, encoding="utf-8")
    (res / "table.md").write_text(tbl, encoding="utf-8")
    print(tbl)
    print("flagged before takeover:", {k: v["flagged_before_takeover"] for k, v in s["verifiers"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
