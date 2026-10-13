"""Figures for the audited-rewards note (simulation part)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bandit_sim import FIG, MU, OUT

COL = {"oracle": "#111827", "judge": "#dc2626", "noisecorr": "#f59e0b", "audit0": "#6b7280", "audited": "#1d4ed8"}
NAME = {"oracle": "gold on every sample", "judge": "judge only", "noisecorr": "fixed-rate noise correction",
        "audit0": "audits only (judge ignored)", "audited": "audited reward (ours)"}


def band(ax, x, arr, color, label, ls="-"):
    m = arr.mean(0)
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    ax.plot(x, m, color=color, lw=2, ls=ls, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    d = np.load(OUT / "sim_curves.npz")
    steps = np.arange(d["oracle_true"].shape[1])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ["oracle", "audited", "audit0", "noisecorr", "judge"]:
        band(ax, steps, d[f"{m}_true"], COL[m], NAME[m])
    ax.plot(steps, d["judge_judge"].mean(0), color=COL["judge"], lw=1.5, ls="--", label="judge only: reward it reports")
    ax.axhline(MU.max(), color="#9ca3af", lw=0.8, ls=":")
    ax.set_xlabel("policy-gradient step")
    ax.set_ylabel("true success rate of the policy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_sim_curves.png", dpi=200); fig.savefig(FIG / "fig1_sim_curves.pdf")

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    band(ax, steps, d["judge_precision"], COL["judge"], "judge's on-policy precision while training on the judge")
    lam = d["audited_lambda"]
    band(ax, np.arange(lam.shape[1]), lam, COL["audited"], "estimated trust g(1) in the audited run")
    ax.set_xlabel("policy-gradient step")
    ax.set_ylabel("P(correct | judge says yes)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_sim_precision.png", dpi=200); fig.savefig(FIG / "fig2_sim_precision.pdf")

    sw = json.loads((OUT / "sim_sweep.json").read_text())
    rates = sorted({v["p"] for v in sw.values()}, reverse=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for m in ("audit0", "audited"):
        ys = [sw[f"{m}@1/{round(1 / p)}"]["final_true_mean"] for p in rates]
        sds = [sw[f"{m}@1/{round(1 / p)}"]["final_true_sd"] for p in rates]
        labels = [sw[f"{m}@1/{round(1 / p)}"]["gold_labels_mean"] for p in rates]
        ax.errorbar(labels, ys, yerr=sds, color=COL[m], marker="o", capsize=3, lw=2, label=NAME[m])
    ax.set_xscale("log")
    ticks = [sw[f"audited@1/{round(1 / p)}"]["gold_labels_mean"] for p in rates]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}\n(p=1/{round(1 / p)})" for t, p in zip(ticks, rates)])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("gold labels used in training (log scale)")
    ax.set_ylabel("final true success (mean ± sd)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_sim_sweep.png", dpi=200); fig.savefig(FIG / "fig3_sim_sweep.pdf")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
