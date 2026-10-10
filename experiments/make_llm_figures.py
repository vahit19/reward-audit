"""Figures and table for the GRPO runs (results/llm/run_*.json).

    uv run --with matplotlib python experiments/make_llm_figures.py [results/llm]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "llm"
FIG = ROOT / "figures"
THEMES = {
    "light": {"bg": "white", "fg": "#1f2933", "grid": "#d9e2ec", "reward": "#d64545", "truth": "#2f6fc4", "flag": "#7b2cbf"},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d", "reward": "#ff7b72", "truth": "#79c0ff", "flag": "#d2a8ff"},
}
TITLES = {"exact": "Exact verifier (true reward)", "substring": "Naive check: gold number anywhere",
          "judge": "Reference-free LLM judge"}


def smooth(xs: list[float], k: int = 9) -> list[float]:
    h = k // 2
    return [statistics.mean(xs[max(0, i - h): i + h + 1]) for i in range(len(xs))]


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["fg"])
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
        lab.set_color(t["fg"])
    ax.grid(color=t["grid"], linewidth=0.6, alpha=0.7)


def main() -> int:
    runs = {}
    for f in sorted(SRC.glob("run_*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(d["source"], []).append(d)
    if not runs:
        print("no runs in", SRC)
        return 1
    FIG.mkdir(exist_ok=True)
    order = [s for s in ("exact", "substring", "judge") if s in runs]
    for name, t in THEMES.items():
        fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.6), facecolor=t["bg"], sharey=True)
        axes = axes if len(order) > 1 else [axes]
        for ax, src in zip(axes, order):
            d = runs[src][0]
            per = {}
            for rec in d["steps"]:
                per.setdefault(rec["step"], []).append(rec)
            steps = sorted(per)
            rew = smooth([statistics.mean(r["reward"] for r in per[k]) for k in steps], 11)
            tru = smooth([statistics.mean(r["true"] for r in per[k]) for k in steps], 11)
            ax.fill_between(steps, tru, rew, where=[r >= t for r, t in zip(rew, tru)], color=t["reward"], alpha=0.12,
                            label="reward paid for wrong answers")
            ax.plot(steps, rew, color=t["reward"], lw=2.2, label="reward the trainer sees")
            ax.plot(steps, tru, color=t["truth"], lw=2.2, label="true accuracy")
            if d.get("flagged_at") is not None:
                ax.axvline(d["flagged_at"], color=t["flag"], ls="--", lw=1.4, label=f"audit flag (step {d['flagged_at']})")
            ax.set_title(TITLES.get(src, src), fontsize=10)
            ax.set_xlabel("GRPO step (32 completions each)")
            ax.set_ylim(0, 0.9)
            style(ax, t)
            leg = ax.legend(fontsize=7.2, facecolor=t["bg"], edgecolor=t["grid"], loc="upper left")
            for txt in leg.get_texts():
                txt.set_color(t["fg"])
        axes[0].set_ylabel("rate (11-step moving average)")
        fig.tight_layout()
        fig.savefig(FIG / f"fig4_grpo_reward_vs_truth_{name}.png", dpi=160, facecolor=t["bg"])
        plt.close(fig)

    lines = ["| Reward source | Reward (first -> last 10 steps) | True accuracy (first -> last 10) | False-reward rate (last 10) | Audit flag step | Audit labels |",
             "|---|---|---|---|---|---|"]
    for src in order:
        for d in runs[src]:
            calls = 4 * 10  # 10 optimizer steps of 4 reward calls each
            last = d["steps"][-calls:]
            first = d["steps"][:calls]
            m = lambda xs, k: statistics.mean(s[k] for s in xs)  # noqa: E731
            lines.append(
                f"| {src} (seed {d['seed']}) | {m(first, 'reward'):.2f} -> {m(last, 'reward'):.2f} | "
                f"{m(first, 'true'):.2f} -> {m(last, 'true'):.2f} | {m(last, 'false_reward'):.3f} | "
                f"{d.get('flagged_at') if d.get('flagged_at') is not None else '-'} | {d.get('audited', '-')} |")
    (SRC / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
