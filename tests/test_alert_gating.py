"""Alert-gating regression: no alert on UNKNOWN primary, citation failure, stale
exchange as-of, or portfolio-cap breach - not just IVP UNKNOWN (7 Nov 2026 decision)."""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nifty_analyst.config import AnalystConfig
from nifty_analyst.pipeline import run_analyst_pipeline
from nifty_analyst.output.audit_log import deliver_run_output
from nifty_analyst.scrubber import IST
from nifty_analyst.state import OpenPosition

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "fixtures"
DEC = dt.datetime(2026, 11, 6, 10, 15, tzinfo=IST)


def _cfg(tmp_path, fixtures=FIXTURES_DIR):
    return AnalystConfig(mode="OFFLINE", fixtures_dir=str(fixtures),
                         audit_log_path=str(tmp_path / "audit.jsonl"))


def test_non_live_ready_suggestion_never_alerts(tmp_path):
    """A SUGGESTION that is not is_live_ready (any cause) must not alert."""
    state, delivery = run_analyst_pipeline(DEC, _cfg(tmp_path))
    assert state.final_output is not None
    if state.final_output.verdict == "SUGGESTION" and state.final_output.is_live_ready:
        # flip the gate flag: delivery must go silent even with a SUGGESTION verdict
        fo = state.final_output
        object.__setattr__(fo, "is_live_ready", False)
        state2 = state
        d2 = deliver_run_output(state2, tmp_path / "audit2.jsonl")
        assert d2.alerted_trader is False
        assert d2.user_facing_message == ""


def test_citation_verification_failure_no_alert(tmp_path):
    """Corrupted synthesis citations -> verifier fails -> WAIT -> silence."""
    from nifty_analyst.synthesis.llm_node import OfflineDeterministicSynthesisProvider

    class CorruptingProvider(OfflineDeterministicSynthesisProvider):
        def synthesize(self, **kwargs):
            resp = super().synthesize(**kwargs)
            for c in resp.citations:
                c["cited_value"] = -999999.0  # contradicts state
            return resp

    state, delivery = run_analyst_pipeline(DEC, _cfg(tmp_path),
                                           llm_provider=CorruptingProvider())
    assert state.final_output is not None
    assert state.final_output.verdict == "WAIT"
    assert delivery.alerted_trader is False
    assert delivery.user_facing_message == ""


def test_stale_chain_quotes_fresh_spot_no_alert(tmp_path):
    """Chain as-of beyond 180s (spot fresh) -> INSUFFICIENT DATA -> silence."""
    fx = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, fx)
    cm = json.loads((fx / "contract_master.json").read_text())
    cm["spot"]["as_of_ist"] = "2026-11-06T10:19:55+05:30"  # fresh at 10:20 decision
    (fx / "contract_master.json").write_text(json.dumps(cm))
    dec = dt.datetime(2026, 11, 6, 10, 20, tzinfo=IST)  # chain quotes 10:14:35 = 325s old
    state, delivery = run_analyst_pipeline(dec, _cfg(tmp_path, fx))
    assert state.final_output is not None
    assert state.final_output.verdict == "INSUFFICIENT DATA"
    assert delivery.alerted_trader is False
    assert delivery.user_facing_message == ""


def test_portfolio_cap_breach_no_alert(tmp_path):
    """Existing positions consuming the Rs 10,000 summed-stop cap -> no qualified primary."""
    positions = [
        OpenPosition(
            position_id=f"pos{i}", structure_type="BULL_PUT_SPREAD", directional_sign=1,
            planned_stop_rs=4000.0, current_executable_spread_value_rs=-1000.0,
            short_leg_deltas=[-0.2], dte_trading_sessions=3,
        )
        for i in range(3)  # 12,000 summed stops > 10,000 cap
    ]
    state, delivery = run_analyst_pipeline(DEC, _cfg(tmp_path), open_positions=positions)
    assert state.final_output is not None
    assert state.final_output.verdict in ("WAIT", "INSUFFICIENT DATA")
    assert delivery.alerted_trader is False
    assert delivery.user_facing_message == ""
