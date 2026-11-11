"""Deterministic expiry-payoff diagram renderer (PNG) for candidate structures.

Pure function of (legs, lot_size, lots, statutory_charges_rs, title): fixed figure
size/dpi/style, Agg backend, no timestamps or randomness. STT is excluded per trader
directive so payoff diagrams show pure contractual expiry P&L net of execution fees.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from nifty_analyst.adapters.request_safety import walk_depth_levels
from nifty_analyst.engine.pricing import (
    LegPayoffSpec,
    compute_piecewise_expiry_payoff,
    evaluate_leg_expiry_pnl_per_unit,
)
from nifty_analyst.state import CandidateStructure

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


def candidate_to_payoff_specs(
    cand: CandidateStructure,
    slippage_buffer_points_per_leg: float = 0.50,
) -> list[LegPayoffSpec]:
    """Build `LegPayoffSpec` entries strictly from the candidate's full-size walked fill prices.

    Never falls back to top-of-book touch quotes when a leg was book-walked (`FULL_SIZE_BOOK_WALK`)
    or lacks full-size fill support (`INDICATIVE_TOUCH_ONLY`).
    """
    if not cand.full_size_fill_supported or cand.execution_fill_basis == "INDICATIVE_TOUCH_ONLY":
        raise ValueError(
            f"PAYOFF_CHART_UNSUPPORTED_FILL: candidate {cand.structure_id!r} lacks full-size "
            f"executable fill support ({cand.execution_fill_basis})."
        )

    slip = float(slippage_buffer_points_per_leg)
    specs: list[LegPayoffSpec] = []
    for leg in cand.legs:
        if leg.fill_basis == "INDICATIVE_TOUCH_ONLY":
            raise ValueError(
                f"PAYOFF_CHART_UNSUPPORTED_LEG_FILL: leg {leg.strike} {leg.option_type} on "
                f"{cand.structure_id!r} is INDICATIVE_TOUCH_ONLY."
            )
        req_units = max(1, cand.lot_size * cand.lots * leg.ratio)
        if leg.executable_entry_price is not None:
            exec_px = float(leg.executable_entry_price)
        elif leg.vwap_fill_price is not None:
            base_px = float(leg.vwap_fill_price)
            exec_px = (
                round(min(base_px, max(0.05, base_px - slip)), 6)
                if leg.side == "SHORT"
                else round(base_px + slip, 6)
            )
        elif leg.book_levels:
            walk_px, _ = walk_depth_levels(leg.book_levels, req_units)
            if walk_px is None:
                raise ValueError(
                    f"PAYOFF_CHART_UNSUPPORTED_DEPTH: leg {leg.strike} {leg.option_type} on "
                    f"{cand.structure_id!r} cannot walk {req_units} units."
                )
            exec_px = (
                round(min(walk_px, max(0.05, walk_px - slip)), 6)
                if leg.side == "SHORT"
                else round(walk_px + slip, 6)
            )
        else:
            if leg.touch_available_qty is not None and leg.touch_available_qty < req_units:
                raise ValueError(
                    f"PAYOFF_CHART_INSUFFICIENT_TOUCH: leg {leg.strike} {leg.option_type} on "
                    f"{cand.structure_id!r} has touch_available_qty={leg.touch_available_qty} < {req_units}."
                )
            base_px = leg.bid if leg.side == "SHORT" else leg.ask
            exec_px = (
                round(min(base_px, max(0.05, base_px - slip)), 6)
                if leg.side == "SHORT"
                else round(base_px + slip, 6)
            )

        specs.append(
            LegPayoffSpec(
                strike=leg.strike,
                option_type=leg.option_type,
                side=leg.side,
                ratio=leg.ratio,
                executable_entry_price=exec_px,
            )
        )
    return specs


def render_candidate_payoff_diagram(
    cand: CandidateStructure,
    out_path: str,
    slippage_buffer_points_per_leg: float = 0.50,
) -> dict:
    """Render a candidate's expiry payoff chart using walked leg prices and verify risk parity."""
    specs = candidate_to_payoff_specs(
        cand, slippage_buffer_points_per_leg=slippage_buffer_points_per_leg
    )
    diag = render_payoff_diagram(
        legs=specs,
        lot_size=cand.lot_size,
        lots=cand.lots,
        statutory_charges_rs=cand.statutory_charges.total_charges_rs,
        title=f"{cand.structure_id} ({cand.structure_type})",
        out_path=out_path,
    )
    expected_net_loss = cand.contractual_max_loss_rs + cand.statutory_charges.total_charges_rs
    be_diag = list(diag["breakevens_net"])
    be_cand = list(cand.breakevens_net)
    be_mismatch = len(be_diag) != len(be_cand) or any(
        abs(float(a) - float(b)) > 0.05 for a, b in zip(be_diag, be_cand)
    )
    if (
        abs(float(diag["gross_max_loss_rs"]) - float(cand.contractual_max_loss_rs)) > 0.05
        or abs(float(diag["max_loss_rs"]) - float(expected_net_loss)) > 0.05
        or abs(float(diag["max_profit_rs"]) - float(cand.max_profit_rs)) > 0.05
        or be_mismatch
    ):
        Path(out_path).unlink(missing_ok=True)
        raise ValueError(
            f"PAYOFF_CHART_RISK_MISMATCH for {cand.structure_id!r}: "
            f"chart=(gross_loss={diag['gross_max_loss_rs']}, net_loss={diag['max_loss_rs']}, "
            f"max_profit={diag['max_profit_rs']}, be={be_diag}) vs "
            f"candidate=(contractual_loss={cand.contractual_max_loss_rs}, "
            f"expected_net_loss={expected_net_loss}, max_profit={cand.max_profit_rs}, be={be_cand})"
        )
    return diag
