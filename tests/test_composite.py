"""Tests for Phase 14: the composite risk view.

Offline tests cover the scoring and combination logic with synthetic fixtures for
FundamentalRisk and ValuationRisk, so the composition rules are pinned without touching the
network or the other two engines. A handful of `integration` tests run `assess()` against a
real ticker, which is the only path that exercises the graceful-degradation behaviour for
real: Module 1 or Module 2 genuinely absent or genuinely failing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.composite import (
    DEFAULT_WEIGHTS,
    REGIME_ADJUSTMENT,
    compose,
    score_market_risk,
    score_valuation_risk,
)
from risk_engine.fundamental import FundamentalRisk
from risk_engine.valuation_risk import ValuationRisk

integration = pytest.mark.integration


def make_fundamental(overall_risk=40.0, deterioration_risk=20.0) -> FundamentalRisk:
    return FundamentalRisk(
        ticker="TEST", name="Test Co", currency="USD", source="SEC EDGAR",
        years=10, first_year=2016, last_year=2025,
        leverage_risk=40.0, liquidity_risk=40.0, earnings_quality_risk=40.0,
        deterioration_risk=deterioration_risk, overall_risk=overall_risk,
        latest={}, trends={}, flags=(), health_score=70.0, health_rating="Healthy",
    )


def make_valuation(terminal_share=0.60, wacc_sensitivity=0.03,
                   wacc=0.09, market_implied_wacc=0.07) -> ValuationRisk:
    return ValuationRisk(
        ticker="TEST", name="Test Co", currency="USD",
        share_price=100.0, implied_share_price=90.0, upside=-0.10,
        wacc=wacc, terminal_growth=0.02, terminal_share=terminal_share,
        market_implied_wacc=market_implied_wacc, market_implied_growth=0.03,
        wacc_sensitivity=wacc_sensitivity, growth_sensitivity=0.04,
        grid=pd.DataFrame(),
    )


def make_market(volatility=0.25, var_99=0.03, max_drawdown=-0.25,
                beta=1.1, beta_confidence="high", regime="normal"):
    return score_market_risk("TEST", volatility, var_99, max_drawdown, beta,
                             beta_confidence, regime)


# --- Market risk scoring ------------------------------------------------------------------------

def test_a_safe_metric_scores_near_zero_and_a_dangerous_one_near_full():
    calm = make_market(volatility=0.10, var_99=0.01, max_drawdown=-0.05, beta=0.7)
    wild = make_market(volatility=0.70, var_99=0.10, max_drawdown=-0.70, beta=3.0)
    assert calm.overall < 15
    assert wild.overall > 85


def test_max_drawdown_is_scored_on_its_magnitude_not_its_sign():
    """drawdown_summary returns a negative number; scoring the raw signed value would treat a
    deep drawdown as safe because it is a large negative number."""
    market = make_market(max_drawdown=-0.55)
    assert market.drawdown_score > 50


def test_a_low_confidence_beta_is_excluded_rather_than_trusted():
    trusted = make_market(beta=2.4, beta_confidence="high")
    excluded = make_market(beta=2.4, beta_confidence="low")
    assert "beta" in excluded.excluded
    assert "beta" not in trusted.excluded
    assert np.isnan(excluded.beta_score)
    assert not np.isnan(trusted.beta_score)


def test_excluding_beta_renormalises_rather_than_penalising():
    """Dropping a metric must not silently drag the score toward either extreme."""
    with_beta = make_market(volatility=0.25, var_99=0.03, max_drawdown=-0.25,
                            beta=1.6, beta_confidence="high")
    without_beta = make_market(volatility=0.25, var_99=0.03, max_drawdown=-0.25,
                               beta=1.6, beta_confidence="low")
    assert abs(with_beta.overall - without_beta.overall) < 25


def test_an_elevated_regime_raises_the_score_by_the_stated_amount():
    normal = make_market(regime="normal")
    elevated = make_market(regime="elevated")
    assert elevated.overall == pytest.approx(
        min(normal.overall + REGIME_ADJUSTMENT, 100.0), abs=0.5)
    assert any("elevated" in n for n in elevated.notes)


def test_a_subdued_regime_lowers_the_score_by_the_stated_amount():
    normal = make_market(regime="normal")
    subdued = make_market(regime="subdued")
    assert subdued.overall == pytest.approx(
        max(normal.overall - REGIME_ADJUSTMENT, 0.0), abs=0.5)


def test_the_regime_adjustment_is_clamped_at_the_boundaries():
    """A metric already at the top of the scale must not be pushed past 100."""
    maxed = make_market(volatility=0.90, var_99=0.12, max_drawdown=-0.90, beta=3.5,
                        regime="elevated")
    assert maxed.overall <= 100.0


# --- Valuation risk scoring -----------------------------------------------------------------------

def test_high_terminal_dependence_and_fragility_score_high_risk():
    fragile = make_valuation(terminal_share=0.85, wacc_sensitivity=0.08)
    score, _ = score_valuation_risk(fragile)
    assert score > 70


def test_low_terminal_dependence_and_stability_score_low_risk():
    stable = make_valuation(terminal_share=0.35, wacc_sensitivity=0.01,
                            wacc=0.09, market_implied_wacc=0.085)
    score, _ = score_valuation_risk(stable)
    assert score < 30


def test_a_missing_market_implied_wacc_is_excluded_and_noted():
    no_solve = make_valuation(market_implied_wacc=None)
    score, notes = score_valuation_risk(no_solve)
    assert not np.isnan(score)
    assert any("No market-implied discount rate" in n for n in notes)


def test_a_wide_gap_to_the_market_implied_rate_raises_the_score():
    close = make_valuation(wacc=0.09, market_implied_wacc=0.088)
    wide = make_valuation(wacc=0.09, market_implied_wacc=0.03)
    close_score, _ = score_valuation_risk(close)
    wide_score, _ = score_valuation_risk(wide)
    assert wide_score > close_score


# --- Composition: weighting, renormalisation and degradation ---------------------------------------

def test_all_three_dimensions_combine_with_the_default_weights():
    market = make_market()
    result = compose("TEST", "Test Co", market, make_fundamental(), make_valuation())
    assert set(result.weights_used) == {"market", "fundamental", "valuation"}
    for key, weight in result.weights_used.items():
        assert weight == pytest.approx(DEFAULT_WEIGHTS[key])
    assert not result.excluded_dimensions


def test_a_missing_fundamental_dimension_is_dropped_and_disclosed():
    market = make_market()
    result = compose("TEST", "Test Co", market, None, make_valuation())
    assert "fundamental" in result.excluded_dimensions
    assert "fundamental" not in result.weights_used
    assert any("Module 1" in n for n in result.notes)


def test_a_missing_valuation_dimension_is_dropped_and_disclosed():
    market = make_market()
    result = compose("TEST", "Test Co", market, make_fundamental(), None)
    assert "valuation" in result.excluded_dimensions
    assert any("Module 2" in n for n in result.notes)


def test_with_only_market_risk_available_the_composite_is_the_market_score():
    market = make_market(volatility=0.30)
    result = compose("TEST", "Test Co", market, None, None)
    assert result.overall == pytest.approx(market.overall)
    assert result.weights_used == {"market": 1.0}


def test_renormalised_weights_still_sum_to_one():
    market = make_market()
    result = compose("TEST", "Test Co", market, make_fundamental(), None)
    assert sum(result.weights_used.values()) == pytest.approx(1.0)


def test_custom_weights_are_respected_when_everything_is_available():
    market = make_market()
    custom = {"market": 0.6, "fundamental": 0.2, "valuation": 0.2}
    result = compose("TEST", "Test Co", market, make_fundamental(), make_valuation(),
                     weights=custom)
    assert result.weights_used["market"] == pytest.approx(0.6)


def test_the_rating_bands_match_the_published_thresholds():
    market = make_market()
    for target, expected in [(20.0, "Low"), (40.0, "Moderate"),
                             (60.0, "Elevated"), (80.0, "High")]:
        fake_fundamental = make_fundamental(overall_risk=target)
        result = compose("TEST", "Test Co", market, fake_fundamental, None,
                         weights={"market": 0.0, "fundamental": 1.0, "valuation": 0.0})
        assert result.overall == pytest.approx(target)
        assert result.rating == expected


# --- Divergences: the actual point of the phase ----------------------------------------------------

def test_fundamental_exceeding_market_is_flagged_as_critical():
    """The dangerous direction: a calm price over a weakening balance sheet."""
    calm_market = make_market(volatility=0.12)  # low market risk
    fragile_business = make_fundamental(overall_risk=85.0)  # high fundamental risk
    result = compose("TEST", "Test Co", calm_market, fragile_business, None)
    kinds = [d.kind for d in result.divergences]
    assert "fundamental_vs_market" in kinds
    match = next(d for d in result.divergences if d.kind == "fundamental_vs_market")
    assert match.severity == "critical"


def test_agreement_between_market_and_fundamental_raises_no_divergence():
    market = make_market(volatility=0.25)
    fundamental = make_fundamental(overall_risk=45.0)
    result = compose("TEST", "Test Co", market, fundamental, None)
    assert not any(d.kind == "fundamental_vs_market" and d.severity == "critical"
                   for d in result.divergences)


def test_fragile_valuation_in_an_elevated_regime_is_flagged():
    elevated_market = make_market(regime="elevated")
    fragile_valuation = make_valuation(wacc_sensitivity=0.08)
    result = compose("TEST", "Test Co", elevated_market, None, fragile_valuation)
    assert any(d.kind == "valuation_vs_regime" for d in result.divergences)


def test_a_stable_valuation_in_an_elevated_regime_is_not_flagged():
    elevated_market = make_market(regime="elevated")
    stable_valuation = make_valuation(wacc_sensitivity=0.01)
    result = compose("TEST", "Test Co", elevated_market, None, stable_valuation)
    assert not any(d.kind == "valuation_vs_regime" for d in result.divergences)


def test_deteriorating_fundamentals_under_a_terminal_heavy_valuation_is_flagged():
    market = make_market()
    deteriorating = make_fundamental(deterioration_risk=75.0)
    terminal_heavy = make_valuation(terminal_share=0.80)
    result = compose("TEST", "Test Co", market, deteriorating, terminal_heavy)
    assert any(d.kind == "deteriorating_vs_terminal" for d in result.divergences)


def test_stable_fundamentals_under_a_terminal_heavy_valuation_is_not_flagged():
    market = make_market()
    stable = make_fundamental(deterioration_risk=15.0)
    terminal_heavy = make_valuation(terminal_share=0.80)
    result = compose("TEST", "Test Co", market, stable, terminal_heavy)
    assert not any(d.kind == "deteriorating_vs_terminal" for d in result.divergences)


def test_critical_divergences_are_filtered_correctly():
    calm_market = make_market(volatility=0.10)
    fragile_business = make_fundamental(overall_risk=90.0)
    result = compose("TEST", "Test Co", calm_market, fragile_business, None)
    assert len(result.critical_divergences) >= 1
    assert all(d.severity == "critical" for d in result.critical_divergences)


def test_a_divergence_renders_its_severity_and_finding():
    market = make_market(volatility=0.10)
    fragile = make_fundamental(overall_risk=90.0)
    result = compose("TEST", "Test Co", market, fragile, None)
    rendered = result.critical_divergences[0].render()
    assert rendered.startswith("[CRITICAL]")


def test_summary_reports_the_rating_and_dimension_count():
    market = make_market()
    result = compose("TEST", "Test Co", market, make_fundamental(), make_valuation())
    text = result.summary()
    assert result.rating in text
    assert "market" in text and "fundamental" in text and "valuation" in text


# --- Integration: assess() against real data --------------------------------------------------------

@integration
def test_assess_produces_a_full_composite_for_a_well_covered_ticker():
    from risk_engine.composite import assess
    result = assess("AAPL")
    assert result.ticker == "AAPL"
    assert 0 <= result.overall <= 100
    assert result.market.overall >= 0


@integration
def test_assess_degrades_gracefully_for_an_index_with_no_filings():
    """An index has no SEC filings and no EBITDA to run a DCF on, so both optional
    dimensions must fall away and the composite must still be the market score alone."""
    from risk_engine.composite import assess
    result = assess("^GSPC")
    assert result.fundamental is None
    assert result.valuation is None
    assert "fundamental" in result.excluded_dimensions
    assert "valuation" in result.excluded_dimensions
    assert result.overall == pytest.approx(result.market.overall)
