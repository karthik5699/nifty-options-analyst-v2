"""Payoff-diagram artifact: deterministic STT-free render + hand-verified analysis numbers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nifty_analyst.engine.pricing import LegPayoffSpec, compute_piecewise_expiry_payoff
from nifty_analyst.output.payoff_diagram import render_payoff_diagram

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def test_bull_put_diagram_numbers_and_determinism(tmp_path: Path) -> None:
    legs = [
        LegPayoffSpec(strike=24900, option_type="PE", side="SHORT", ratio=1, executable_entry_price=65.0),
        LegPayoffSpec(strike=24700, option_type="PE", side="LONG", ratio=1, executable_entry_price=35.0),
    ]
    pw = compute_piecewise_expiry_payoff(legs, 65, 1, statutory_charges_rs=93.5)
    assert pw.is_defined_risk is True
    assert abs(pw.max_loss_rs - 11143.5) < 1e-6      # (200-30)*65 + 93.5
    assert abs(pw.max_profit_rs - 1856.5) < 1e-6     # 30*65 - 93.5
    assert abs(pw.breakevens_net[0] - 24871.44) < 0.01

    if not HAS_MATPLOTLIB:
        return

    p1 = str(tmp_path / "a.png")
    p2 = str(tmp_path / "b.png")
    r = render_payoff_diagram(legs, 65, 1, 93.5, "BPS", p1)
    assert r["is_defined_risk"] is True
    assert abs(r["max_loss_rs"] - 11143.5) < 1e-6
    assert abs(r["max_profit_rs"] - 1856.5) < 1e-6
    render_payoff_diagram(legs, 65, 1, 93.5, "BPS", p2)
    b1, b2 = Path(p1).read_bytes(), Path(p2).read_bytes()
    assert b1[:8] == b"\x89PNG\r\n\x1a\n"
    assert b1 == b2, "renderer must be deterministic for identical inputs"


def test_debit_call_diagram_is_bounded_without_stt(tmp_path: Path) -> None:
    legs = [
        LegPayoffSpec(strike=25000, option_type="CE", side="LONG", ratio=1, executable_entry_price=120.0),
        LegPayoffSpec(strike=25200, option_type="CE", side="SHORT", ratio=1, executable_entry_price=55.0),
    ]
    pw = compute_piecewise_expiry_payoff(legs, 65, 1, statutory_charges_rs=93.5)
    assert pw.is_defined_risk is True
    assert pw.method_description == "LEG_WISE_PIECEWISE_EXPIRY_PAYOFF"
    assert abs(pw.max_loss_rs - 4318.5) < 1e-6       # (120-55)*65 + 93.5
    assert abs(pw.gross_max_loss_rs - 4225.0) < 1e-6  # (120-55)*65

    if not HAS_MATPLOTLIB:
        return

    r = render_payoff_diagram(legs, 65, 1, 93.5, "BCS", str(tmp_path / "c.png"))
    assert r["method_description"] == "LEG_WISE_PIECEWISE_EXPIRY_PAYOFF"
    assert abs(r["max_loss_rs"] - 4318.5) < 1e-6
