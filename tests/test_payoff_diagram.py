"""Payoff-diagram artifact: deterministic render + hand-verified analysis numbers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nifty_analyst.engine.pricing import LegPayoffSpec
from nifty_analyst.output.payoff_diagram import render_payoff_diagram


def test_bull_put_diagram_numbers_and_determinism(tmp_path):
    legs = [
        LegPayoffSpec(strike=24900, option_type="PE", side="SHORT", ratio=1, executable_entry_price=65.0),
        LegPayoffSpec(strike=24700, option_type="PE", side="LONG", ratio=1, executable_entry_price=35.0),
    ]
    p1 = str(tmp_path / "a.png")
    p2 = str(tmp_path / "b.png")
    r = render_payoff_diagram(legs, 65, 1, 93.5, 0.0015, "t", p1)
    assert r["is_defined_risk"] is True
    assert abs(r["max_loss_rs"] - 11143.5) < 1e-6      # (200-30)*65 + 93.5
    assert abs(r["max_profit_rs"] - 1856.5) < 1e-6     # 30*65 - 93.5
    assert abs(r["breakevens_net"][0] - 24871.44) < 0.01
    render_payoff_diagram(legs, 65, 1, 93.5, 0.0015, "t", p2)
    b1, b2 = Path(p1).read_bytes(), Path(p2).read_bytes()
    assert b1[:8] == b"\x89PNG\r\n\x1a\n"
    assert b1 == b2, "renderer must be deterministic for identical inputs"


def test_debit_call_diagram_flags_expiry_hold_stt_tail(tmp_path):
    legs = [
        LegPayoffSpec(strike=25000, option_type="CE", side="LONG", ratio=1, executable_entry_price=120.0),
        LegPayoffSpec(strike=25200, option_type="CE", side="SHORT", ratio=1, executable_entry_price=55.0),
    ]
    r = render_payoff_diagram(legs, 65, 1, 93.5, 0.0015, "t", str(tmp_path / "c.png"))
    assert "EXPIRY_HOLD_UNBOUNDED_EXERCISE_STT" in r["method_description"]
    assert abs(r["max_loss_rs"] - 4318.5) < 1e-6       # (120-55)*65 + 93.5
