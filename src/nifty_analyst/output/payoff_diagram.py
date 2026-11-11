"""Deterministic expiry-payoff diagram renderer (PNG) for candidate structures.

Pure function of (legs, lot_size, lots, statutory_charges_rs, title): fixed figure
size/dpi/style, Agg backend, no timestamps or randomness. STT is excluded per trader
directive so payoff diagrams show pure contractual expiry P&L net of execution fees.
"""

from __future__ import annotations

import math
from typing import Sequence

from nifty_analyst.engine.pricing import (
    LegPayoffSpec,
    compute_piecewise_expiry_payoff,
    evaluate_leg_expiry_pnl_per_unit,
)

FIGSIZE = (8.0, 4.5)
DPI = 120
SPOT_PAD_PCT = 0.08
GRID_POINTS = 801


def _net_pnl_points(legs: Sequence[LegPayoffSpec], s_t: float) -> float:
    return sum(evaluate_leg_expiry_pnl_per_unit(leg, s_t) for leg in legs)


def _side_label(legs: Sequence[LegPayoffSpec]) -> str:
    parts = []
    for leg in legs:
        side = "short" if leg.side == "SHORT" else "long"
        ratio_prefix = f"{leg.ratio}x " if leg.ratio > 1 else ""
        parts.append(f"{side} {ratio_prefix}{leg.strike:,.0f} {leg.option_type}")
    return " / ".join(parts)


def render_payoff_diagram(
    legs: Sequence[LegPayoffSpec],
    lot_size: int,
    lots: int,
    statutory_charges_rs: float,
    title: str,
    out_path: str,
) -> dict:
    """Render the NET expiry payoff curve to PNG; returns the analysis dict for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = compute_piecewise_expiry_payoff(
        legs,
        lot_size,
        lots,
        statutory_charges_rs=statutory_charges_rs,
    )
    qty = lot_size * lots
    strikes = sorted({leg.strike for leg in legs})
    lo = min(strikes) * (1.0 - SPOT_PAD_PCT)
    hi = max(strikes) * (1.0 + SPOT_PAD_PCT)
    if math.isinf(analysis.max_loss_rs):
        hi = max(strikes) * 1.6

    grid = [lo + (hi - lo) * i / (GRID_POINTS - 1) for i in range(GRID_POINTS)]
    net_rs = [_net_pnl_points(legs, s) * qty - statutory_charges_rs for s in grid]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("#12161c")
    ax.set_facecolor("#12161c")
    ax.plot(grid, net_rs, color="#4da3ff", linewidth=1.8, label="Net expiry P&L (Rs)")
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    for k in strikes:
        ax.axvline(k, color="#3a4250", linewidth=0.6, linestyle="--")
    for be in analysis.breakevens_net:
        if lo <= be <= hi:
            ax.plot([be], [0.0], marker="o", markersize=5, color="#ff7b72")
            ax.annotate(
                f"BE {be:,.0f}",
                (be, 0.0),
                textcoords="offset points",
                xytext=(4, 8),
                fontsize=8,
                color="#ff7b72",
            )
    if not math.isinf(analysis.max_loss_rs):
        ax.axhline(
            -analysis.max_loss_rs,
            color="#ff7b72",
            linewidth=0.9,
            linestyle=":",
            label=f"Max loss Rs {analysis.max_loss_rs:,.0f}",
        )
    if not math.isinf(analysis.max_profit_rs):
        ax.axhline(
            analysis.max_profit_rs,
            color="#7bd88f",
            linewidth=0.9,
            linestyle=":",
            label=f"Max profit Rs {analysis.max_profit_rs:,.0f}",
        )
    ax.set_title(f"{title} [{_side_label(legs)}]", fontsize=10)
    ax.set_xlabel("NIFTY spot at expiry")
    ax.set_ylabel("P&L (Rs, net of execution fees)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, metadata={"Software": "nifty-analyst payoff_diagram v2"})
    plt.close(fig)
    return {
        "is_defined_risk": analysis.is_defined_risk,
        "max_loss_rs": analysis.max_loss_rs,
        "gross_max_loss_rs": analysis.gross_max_loss_rs,
        "max_profit_rs": analysis.max_profit_rs,
        "breakevens_net": analysis.breakevens_net,
        "method_description": analysis.method_description,
        "out_path": out_path,
    }
