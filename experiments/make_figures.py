"""Figures for the controlled simulation (light and dark variants).

    uv run --with matplotlib python experiments/make_figures.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reward_audit.monitor import FalseRewardMonitor  # noqa: E402
from reward_audit.train import run  # noqa: E402

FIG = ROOT / "figures"
SEEDS = list(range(40))
THEMES = {
    "light": {"bg": "white", "fg": "#1f2933", "grid": "#d9e2ec", "reward": "#d64545", "truth": "#2f6fc4",
              "flag": "#7b2cbf", "take": "#e8590c", "mix": ["#2f6fc4", "#e8590c", "#f0a202"]},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d", "reward": "#ff7b72", "truth": "#79c0ff",
             "flag": "#d2a8ff", "take": "#ffa657", "mix": ["#79c0ff", "#ffa657", "#e3b341"]},
}


def _style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["fg"])
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])
    ax.grid(color=t["grid"], linewidth=0.6, alpha=0.7)


def _median_curve(runs, attr):
    curves = [getattr(r, attr) for r in runs]
    return [statistics.median(c[i] for c in curves) for i in range(len(curves[0]))]


def hero(runs, t, name):
    reward, truth = _median_curve(runs, "reward_curve"), _median_curve(runs, "true_curve")
    flag = statistics.median(r.flagged_at for r in runs if r.flagged_at is not None)
    take = statistics.median(r.takeover_step for r in runs if r.takeover_step is not None)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=t["bg"])
    ax.plot(reward, color=t["reward"], lw=2.2, label="reward the trainer sees")
    ax.plot(truth, color=t["truth"], lw=2.2, label="true accuracy")
    ax.axvline(flag, color=t["flag"], ls="--", lw=1.6, label=f"audit flags the reward (step {flag:g})")
    ax.axvline(take, color=t["take"], ls=":", lw=1.8, label=f"exploit takes over the policy (step {take:g})")
    ax.set_xlabel("training step")
    ax.set_ylabel("rate (median of 40 runs)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("A lenient verifier is exploited; 2 audit labels per step catch it first", fontsize=11)
    _style(ax, t)
    leg = ax.legend(loc="center right", fontsize=8.5, facecolor=t["bg"], edgecolor=t["grid"])
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout()
    fig.savefig(FIG / f"fig1_reward_vs_truth_{name}.png", dpi=160, facecolor=t["bg"])
    plt.close(fig)


def mix(run0, t, name):
    from reward_audit.policy import Policy  # noqa: F401  (documented strategies)

    exploit = run0.exploit_curve
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=t["bg"])
    steps = range(len(exploit))
    ax.stackplot(steps, [1 - e for e in exploit], exploit, colors=[t["mix"][0], t["mix"][1]],
                 labels=["honest solve", "exploit (spray + bluff)"], alpha=0.9)
    ax.set_xlabel("training step")
    ax.set_ylabel("policy share")
    ax.set_ylim(0, 1)
    ax.set_title("Strategy mix under the lenient verifier (seed 0)", fontsize=11)
    _style(ax, t)
    leg = ax.legend(loc="center left", fontsize=8.5, facecolor=t["bg"], edgecolor=t["grid"])
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout()
    fig.savefig(FIG / f"fig2_strategy_mix_{name}.png", dpi=160, facecolor=t["bg"])
    plt.close(fig)


def labels_needed(t, name):
    import random

    rates = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]
    med = []
    for p in rates:
        needed = []
        for s in range(200):
            rng = random.Random(s)
            m = FalseRewardMonitor(eps=0.02, alpha=0.05)
            for i in range(4000):
                if m.update(int(rng.random() < p), i):
                    needed.append(i + 1)
                    break
            else:
                needed.append(4000)
        med.append(statistics.median(needed))
    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=t["bg"])
    ax.plot([r * 100 for r in rates], med, marker="o", color=t["flag"], lw=2)
    ax.set_xlabel("false-reward rate (%)  [tolerance: 2%]")
    ax.set_ylabel("audit labels to flag (median)")
    ax.set_yscale("log")
    ax.set_title("How many audited outputs the monitor needs", fontsize=11)
    _style(ax, t)
    fig.tight_layout()
    fig.savefig(FIG / f"fig3_labels_needed_{name}.png", dpi=160, facecolor=t["bg"])
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    runs = [run("substring", s) for s in SEEDS]
    for name, t in THEMES.items():
        hero(runs, t, name)
        mix(runs[0], t, name)
        labels_needed(t, name)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
