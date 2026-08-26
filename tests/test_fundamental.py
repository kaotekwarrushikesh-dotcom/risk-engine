"""Tests for Phase 9: fundamental risk.

Split into offline tests of the scoring and trend logic, and integration tests marked
`integration` that need Module 1 installed and the network. The offline set covers the
reasoning; the integration set checks the seam between the two engines actually holds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.fundamental import (
    COVERAGE_CRITICAL,
    RiskFlag,
    _mean_available,
    _score,
    _slope_per_year,
    compare_with_market_risk,
)

integration = pytest.mark.integration


class FakeFundamental:
    """Minimal stand-in carrying only what compare_with_market_risk reads."""

    def __init__(self, overall):
        self.overall_risk = overall


# --- Scoring ---------------------------------------------------------------------------------

def test_a_safe_value_scores_zero_risk_and_a_dangerous_one_scores_full():
    assert _score(15.0, safe=15.0, dangerous=2.0) == 0.0
    assert _score(2.0, safe=15.0, dangerous=2.0) == 100.0


def test_scoring_is_linear_between_the_bounds():
    assert _score(8.5, safe=15.0, dangerous=2.0) == pytest.approx(50.0)


def test_scores_are_clamped_outside_the_bounds():
    """A company far safer than the safe bound is not negative risk, and one far past the
    dangerous bound is not more than certain."""
    assert _score(40.0, safe=15.0, dangerous=2.0) == 0.0
    assert _score(-5.0, safe=15.0, dangerous=2.0) == 100.0


def test_the_direction_of_the_bounds_sets_which_way_is_risky():
    # Higher debt is riskier: safe below, dangerous above.
    assert _score(3.0, safe=0.3, dangerous=3.0) == 100.0
    assert _score(0.3, safe=0.3, dangerous=3.0) == 0.0


def test_a_missing_value_scores_nan_rather_than_safe():
    """Scoring a data gap as zero risk would invent a clean bill of health."""
    assert np.isnan(_score(float("nan"), 15.0, 2.0))
    assert np.isnan(_score(None, 15.0, 2.0))


def test_averaging_ignores_missing_components_rather_than_treating_them_as_zero():
    assert _mean_available([80.0, float("nan"), 40.0]) == pytest.approx(60.0)
    assert np.isnan(_mean_available([float("nan"), float("nan")]))


# --- Trend slopes ------------------------------------------------------------------------------

def test_a_falling_series_produces_a_negative_slope():
    assert _slope_per_year(pd.Series([10.0, 8.0, 6.0, 4.0])) == pytest.approx(-2.0)


def test_a_rising_series_produces_a_positive_slope():
    assert _slope_per_year(pd.Series([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)


def test_a_flat_series_has_no_slope():
    assert _slope_per_year(pd.Series([5.0] * 5)) == pytest.approx(0.0)


def test_the_slope_uses_every_year_not_just_the_endpoints():
    """A single unusual endpoint must not decide whether a company is called deteriorating."""
    steady_decline = pd.Series([10.0, 8.0, 6.0, 4.0, 2.0])
    spike_at_the_end = pd.Series([10.0, 10.0, 10.0, 10.0, 2.0])

    endpoints_are_identical = (
        steady_decline.iloc[0] == spike_at_the_end.iloc[0]
        and steady_decline.iloc[-1] == spike_at_the_end.iloc[-1])
    assert endpoints_are_identical
    assert _slope_per_year(steady_decline) != pytest.approx(_slope_per_year(spike_at_the_end))


def test_too_few_points_gives_no_slope_rather_than_a_two_point_line():
    assert np.isnan(_slope_per_year(pd.Series([1.0, 2.0])))


def test_missing_years_are_dropped_before_fitting():
    assert _slope_per_year(pd.Series([10.0, np.nan, 6.0, 4.0])) == pytest.approx(-3.0)


# --- Flags ---------------------------------------------------------------------------------------

def test_a_flag_renders_its_severity_category_and_evidence():
    flag = RiskFlag("Leverage", "critical", "Interest cover is thin",
                    "EBIT covers interest 1.2x", "A downturn threatens debt service.")
    rendered = flag.render()
    assert "CRITICAL" in rendered and "Leverage" in rendered and "1.2x" in rendered


# --- Market versus fundamental risk, which is the point of the phase --------------------------------

def test_a_fragile_business_with_a_calm_share_price_is_named_as_such():
    """The dangerous direction: volatility cannot see leverage building under a quiet stock."""
    result = compare_with_market_risk(FakeFundamental(80.0), annualised_volatility=0.16)
    assert result["verdict"] == "fundamental risk exceeds market risk"
    assert result["gap"] > 25
    assert "calm share price is not evidence about the balance sheet" in result["reading"]


def test_a_solid_business_with_a_volatile_share_price_is_named_as_such():
    result = compare_with_market_risk(FakeFundamental(20.0), annualised_volatility=0.65)
    assert result["verdict"] == "market risk exceeds fundamental risk"
    assert result["gap"] < -25


def test_agreement_is_reported_when_the_two_measures_broadly_match():
    result = compare_with_market_risk(FakeFundamental(45.0), annualised_volatility=0.33)
    assert result["verdict"] == "the two measures broadly agree"
    assert abs(result["gap"]) <= 25


def test_a_peer_volatility_changes_the_comparison_basis():
    """Against a volatile peer group, the same volatility is unremarkable."""
    absolute = compare_with_market_risk(FakeFundamental(50.0), 0.45)
    relative = compare_with_market_risk(FakeFundamental(50.0), 0.45, peer_volatility=0.45)
    assert relative["market_risk"] < absolute["market_risk"]
    assert "peer volatility" in relative["basis"]
    assert "cruder comparison" in absolute["basis"]


def test_the_comparison_basis_is_always_disclosed():
    for peers in (None, 0.3):
        assert compare_with_market_risk(FakeFundamental(50.0), 0.3, peers)["basis"]


# --- Integration with Module 1 ------------------------------------------------------------------------

@integration
def test_a_real_company_produces_a_scored_risk_profile():
    from risk_engine.fundamental import assess

    result = assess("AAPL")
    assert result.ticker == "AAPL"
    assert 0 <= result.overall_risk <= 100
    assert result.rating in ("Low", "Moderate", "Elevated", "High")
    assert result.years >= 5


@integration
def test_ratios_come_from_module_1_rather_than_being_recomputed():
    """The health score travelling with the result is the evidence that Module 1 ran, rather
    than this module having reimplemented the ratios."""
    from risk_engine.fundamental import assess

    result = assess("AAPL")
    assert result.health_score is not None
    assert result.health_rating is not None
    assert result.source


@integration
def test_deterioration_is_detected_where_it_genuinely_occurred():
    """Pfizer's post-pandemic decline is a real, documented deterioration in the accounts, so
    it is the right case to check the slope logic against something that actually happened."""
    from risk_engine.fundamental import assess

    result = assess("PFE")
    assert result.deterioration_risk > 50
    assert any(f.category == "Deterioration" for f in result.flags)


@integration
def test_missing_metrics_are_excluded_and_disclosed_rather_than_scored_safe():
    from risk_engine.fundamental import assess

    result = assess("AAPL")
    if result.unavailable:
        assert any("excluded rather than scored" in n for n in result.notes)


@integration
def test_a_nonexistent_company_raises_rather_than_returning_a_clean_profile():
    from risk_engine.fundamental import assess

    with pytest.raises(Exception):
        assess("NOT_A_REAL_TICKER_XYZQ")
