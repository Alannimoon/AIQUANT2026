"""Figure 5 — Per-round IC evolution of LLM-based alpha miners.

The AlphaAgent paper's Figure 5 shows the *mean* IC across 5 iterative
rounds for AlphaAgent / RD-Agent / AlphaForge, with shaded ±1/2 std
regions over 20 independent trials. The paper claims AlphaAgent's
expanding shaded region demonstrates `exploration of the factor space'.

Our reproduction tells a different — and, for ELITEALPHA, more useful —
story: with our setup (DeepSeek + chenditc CSI500), AlphaAgent's per-round
IC is essentially flat across 5 rounds, with all post-loop-0 factors
sharing the same core formula (cumulative_return / volatility) and the
same IC to four decimal places. This is direct evidence of the `factor
crowding' / attractor-collapse problem that motivates ELITEALPHA's
MAP-Elites quality-diversity search.

When SYS provides EliteAlpha's per-round IC sequence, add it here as a
second line — the contrast (flat AlphaAgent vs rising/diversifying
EliteAlpha) is the figure's payoff.

Usage:
    python scripts/plot_figure5.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Per-loop SOTA-vs-current IC from this morning's local mine
# (run_logs/alphaagent_table2.log, see "Combined Results" blocks).
# Loop 0's `Current` factor became the SOTA — the flat 0.016563 line
# afterwards is the "core formula + decoration" attractor.
ALPHAAGENT_OURS = {
    "rounds": [1, 2, 3, 4, 5],
    "ic":     [0.016417, 0.016563, 0.016563, 0.016563, 0.016563],
    "factors": [
        "ZSCORE(TS_SUM($return, 5))",
        "TS_SUM($return, 5) / TS_STD($return, 20)",
        "TS_SUM($return, 10) / TS_STD($return, 20)",
        "TS_SUM($return, 10) / TS_STD($return, 5)",
        "RANK(TS_SUM($return, 10) / TS_STD($return, 5))",
    ],
}

# Placeholder. SYS will provide real per-round mean IC + std once the
# MAP-Elites archive is implemented. Until then, draw nothing for it.
ELITEALPHA_OURS = None  # {"rounds": [...], "ic_mean": [...], "ic_std": [...]}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    rounds = ALPHAAGENT_OURS["rounds"]
    ic = ALPHAAGENT_OURS["ic"]
    ax.plot(rounds, ic, color="#e41a1c", linewidth=2.2, marker="o",
            markersize=8, label="AlphaAgent (ours, DeepSeek + CSI500)")

    # Annotate each round with the factor formula's "core".
    cores = [
        "$\\Sigma$ return₅",
        "$\\Sigma$ ret₅ / std₂₀",
        "$\\Sigma$ ret₁₀ / std₂₀",
        "$\\Sigma$ ret₁₀ / std₅",
        "RANK($\\Sigma$ ret₁₀ / std₅)",
    ]
    for r, y, c in zip(rounds, ic, cores):
        ax.annotate(c, xy=(r, y), xytext=(0, 12), textcoords="offset points",
                    fontsize=8, ha="center", color="#555")

    # Optional ELITEALPHA line.
    if ELITEALPHA_OURS is not None:
        rr = ELITEALPHA_OURS["rounds"]
        m = ELITEALPHA_OURS["ic_mean"]
        s = ELITEALPHA_OURS["ic_std"]
        ax.plot(rr, m, color="#000000", linewidth=2.5, marker="D",
                markersize=8, label="EliteAlpha (ours, MAP-Elites)")
        ax.fill_between(rr, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)],
                        color="#000000", alpha=0.18, label="EliteAlpha ±1 std")

    ax.set_xlabel("Round")
    ax.set_ylabel("IC")
    ax.set_xticks(rounds)
    ax.set_ylim(0.014, 0.020)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_title("Figure 5: Per-round IC of AlphaAgent's mining loop on CSI 500\n"
                 "(flat trajectory $\\Rightarrow$ attractor collapse — motivates ELITEALPHA)",
                 fontsize=11)

    fig.tight_layout()
    out_pdf = FIG_DIR / "figure5_csi500.pdf"
    out_png = FIG_DIR / "figure5_csi500.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"       {out_png}")


if __name__ == "__main__":
    main()
