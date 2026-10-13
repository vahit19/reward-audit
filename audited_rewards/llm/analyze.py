"""Compare GRPO runs: exact / judge (from reward-audit) and audited / audit0 (this project)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NEW = HERE.parent / "results" / "llm"
OLD = HERE.parents[1].parent / "results" / "llm"
FIG = HERE.parent / "paper"
RUNS = {"exact": OLD / "run_exact_0.json", "judge": OLD / "run_judge_0.json",
        "audited": NEW / "run_audited_0.json", "audit0": NEW / "run_audit0_0.json"}
NAME = {"exact": "gold on every sample (32/step)", "judge": "judge only (0 gold)",
        "audited": "audited reward (4 gold/step)", "audit0": "audits only, judge ignored (4 gold/step)"}
COL = {"exact": "#111827", "judge": "#dc2626", "audited": "#1d4ed8", "audit0": "#6b7280"}


def per_step(log: dict, key: str) -> list[float]:
    by = {}
    for r in log["steps"]:
        if key in r:
            by.setdefault(r["step"], []).append(r[key])
    return [sum(v) / len(v) for _, v in sorted(by.items())]


def smooth(xs: list[float], k: int = 11) -> list[float]:
    h = k // 2
    return [sum(xs[max(0, i - h): i + h + 1]) / len(xs[max(0, i - h): i + h + 1]) for i in range(len(xs))]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    logs = {k: json.loads(p.read_text()) for k, p in RUNS.items() if p.exists()}
    rows = []
    for k, log in logs.items():
        tru, rew = per_step(log, "true"), per_step(log, "reward")
        row = {"source": k, "true_first10": mean(tru[:10]), "true_last10": mean(tru[-10:]),
               "true_last50": mean(tru[-50:]), "reward_last10": mean(rew[-10:]), "reward_mean_all": mean(rew),
               "true_mean_all": mean(tru), "flagged_at": log.get("flagged_at"), "minutes": log.get("minutes")}
        if "judge" in log["steps"][0]:
            row["judge_last10"] = mean(per_step(log, "judge")[-10:])
            g1 = per_step(log, "g1")
            row["g1_first10"], row["g1_last10"] = mean(g1[:10]), mean(g1[-10:])
        rows.append(row)
    (NEW / "comparison.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for k, log in logs.items():
        axes[0].plot(smooth(per_step(log, "true")), color=COL[k], lw=2, label=NAME[k])
        axes[1].plot(smooth(per_step(log, "reward")), color=COL[k], lw=2)
    for k in ("judge", "audited"):
        if k in logs:
            axes[1].plot(smooth(per_step(logs[k], "true")), color=COL[k], lw=1.2, ls=":")
    if "audited" in logs:
        a = logs["audited"]
        axes[2].plot(smooth(per_step(a, "judge")), color=COL["judge"], lw=2, label="judge says correct")
        axes[2].plot(smooth(per_step(a, "true")), color="#111827", lw=2, label="actually correct")
        axes[2].plot(smooth(per_step(a, "g1")), color=COL["audited"], lw=2, label="trust $g$(yes) from audits")
    axes[0].set_title("(a) true accuracy of the policy", fontsize=10)
    axes[1].set_title("(b) reward the trainer sees\n(dotted: true accuracy of that run)", fontsize=10)
    axes[2].set_title("(c) audited run: judge vs. truth", fontsize=10)
    for ax in axes:
        ax.set_xlabel("GRPO step")
        ax.set_ylim(0, 0.8)
    axes[2].set_ylim(0, 1.0)
    axes[2].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig4_grpo.png", dpi=200); fig.savefig(FIG / "fig4_grpo.pdf")
    print("figure written")


if __name__ == "__main__":
    main()
