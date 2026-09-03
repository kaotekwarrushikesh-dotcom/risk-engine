"""Phase 14: one risk view, built from all three engines.

Phases 1 to 8 answer "how much does the price move" (market risk). Phase 9 answers "how
fragile is the business" (fundamental risk). Phase 10 answers "how much does the valuation
depend on its assumptions" (valuation risk). Nothing so far puts the three side by side and
says what to make of them together. That is this phase.

**A composite score is not a fourth opinion, it is the same three numbers on one scale.**
Each dimension already produces something in the 0-100, "higher is riskier" convention this
engine uses throughout (Module 1's health score, Phase 9's fundamental risk). This module
reads those, maps valuation risk onto the same scale, and combines them the way Module 1
combines its pillars: weighted, and renormalised over whatever is actually available rather
than penalised for what is missing. A ticker with no SEC filings still gets a market-risk
score; it just does not get a fundamental one, and the result says so.

**The reason to build this at all is the disagreements, not the average.** A single number
that blends three independent readings is the least informative thing this phase can produce;
the useful part is naming *where* they disagree, because each disagreement is a different
warning:

  Fundamental > market    A quiet share price sitting over a weakening balance sheet. Price
                          volatility cannot see leverage building, so the calm is not
                          evidence the accounts do not support (Phase 9's own finding,
                          reused here rather than re-derived).
  Valuation fragile,      The inputs a DCF's sensitivity depends on (beta, discount rate)
  market regime elevated  are least stable exactly when the valuation is most sensitive to
                          them, which is the worst time to trust a point estimate.
  Fundamental             Most of the valuation rests on a perpetuity assumption about a
  deteriorating, high     business whose own recent trend is the assumption's biggest risk:
  terminal dependence     growing more slowly, or not at all.

Every function that scores or combines is pure and takes already-computed inputs, so it can
be tested without a network call. Only `assess()`, which fetches live data and calls Phases 9
and 10, needs one.
"""

from dataclasses import dataclass, field

import numpy as np

from risk_engine.fundamental import FundamentalRisk, compare_with_market_risk
from risk_engine.valuation_risk import ValuationRisk

# Market-risk metric bounds: safe scores 0, dangerous scores 100. Volatility and drawdown
# bounds match the fallback used in Phase 9's own market-risk comparison, so the two stay on
# one consistent scale rather than each phase inventing its own idea of "high".
VOLATILITY_SAFE, VOLATILITY_DANGEROUS = 0.15, 0.55
VAR_SAFE, VAR_DANGEROUS = 0.02, 0.08
DRAWDOWN_SAFE, DRAWDOWN_DANGEROUS = 0.15, 0.60
BETA_SAFE, BETA_DANGEROUS = 0.8, 2.5

# Valuation-risk metric bounds, same convention.
TERMINAL_SHARE_SAFE, TERMINAL_SHARE_DANGEROUS = 0.40, 0.85
WACC_SENSITIVITY_SAFE, WACC_SENSITIVITY_DANGEROUS = 0.02, 0.08
IMPLIED_WACC_GAP_SAFE, IMPLIED_WACC_GAP_DANGEROUS = 0.01, 0.05

# How far a GARCH regime nudges the market score once the other metrics have set it. Bounded
# deliberately small: the regime is context for the four measured metrics, not a fifth one.
REGIME_ADJUSTMENT = 8.0

DEFAULT_WEIGHTS = {"market": 0.40, "fundamental": 0.30, "valuation": 0.30}


def _score(value: float | None, safe: float, dangerous: float) -> float:
    """Linear interpolation onto 0-100, `safe` at 0 and `dangerous` at 100, clamped outside.

    The same mapping Phase 9 uses, kept as its own small copy here rather than imported: each
    scoring module owns its scale rather than reaching into another's private helper.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    span = dangerous - safe
    if span == 0:
        return float("nan")
    return float(np.clip((value - safe) / span, 0.0, 1.0) * 100.0)


def _weighted_average(scored: list[tuple[float, float]]) -> float:
    """Mean of (score, weight) pairs, dropping NaNs and renormalising what remains.

    A metric that cannot be computed is excluded rather than scored as safe or as maximally
    risky; either would fabricate a reading. This is the same rule Module 1's health score
    and Phase 9's fundamental risk both apply, kept consistent across the platform.
    """
    usable = [(s, w) for s, w in scored if not np.isnan(s)]
    if not usable:
        return float("nan")
    total_weight = sum(w for _, w in usable)
    return sum(s * w for s, w in usable) / total_weight


@dataclass(frozen=True)
class MarketRiskScore:
    """Phases 2-8's metrics, reduced to one 0-100 reading on the same scale as the others."""

    ticker: str
    volatility: float
    var_99: float
    max_drawdown: float
    beta: float | None
    beta_confidence: str | None
    regime: str

    volatility_score: float
    var_score: float
    drawdown_score: float
    beta_score: float

    overall: float
    excluded: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{self.ticker}: market risk {self.overall:.0f}/100. Volatility {self.volatility:.1%}, "
            f"99% VaR {self.var_99:.2%}, max drawdown {self.max_drawdown:.1%}, "
            f"regime {self.regime}."
        )


def score_market_risk(
    ticker: str,
    volatility: float,
    var_99: float,
    max_drawdown: float,
    beta: float | None = None,
    beta_confidence: str | None = None,
    regime: str = "normal",
) -> MarketRiskScore:
    """Combine volatility, tail loss, drawdown and beta into one market-risk reading.

    `max_drawdown` is signed (negative); its magnitude is what is scored, matching the
    convention `drawdown_summary` already returns. A low-confidence beta is excluded rather
    than trusted at face value, for the same reason Phase 3 itself downgrades one: an
    R-squared near zero means the number looks meaningful and mostly is not.
    """
    excluded: list[str] = []
    notes: list[str] = []

    volatility_score = _score(volatility, VOLATILITY_SAFE, VOLATILITY_DANGEROUS)
    var_score = _score(var_99, VAR_SAFE, VAR_DANGEROUS)
    drawdown_score = _score(abs(max_drawdown), DRAWDOWN_SAFE, DRAWDOWN_DANGEROUS)

    if beta is None or beta_confidence == "low":
        beta_score = float("nan")
        if beta is not None:
            excluded.append("beta")
            notes.append(
                f"Beta ({beta:.2f}) is excluded from the market score because its confidence "
                "is low: an R-squared near zero means the reading looks meaningful and "
                "mostly is not, the same standard Phase 3 itself applies."
            )
    else:
        beta_score = _score(beta, BETA_SAFE, BETA_DANGEROUS)

    overall = _weighted_average([
        (volatility_score, 0.30), (var_score, 0.30),
        (drawdown_score, 0.25), (beta_score, 0.15),
    ])

    if regime == "elevated":
        overall = float(np.clip(overall + REGIME_ADJUSTMENT, 0.0, 100.0))
        notes.append(
            f"Current volatility regime is elevated, which nudges the market score up by "
            f"{REGIME_ADJUSTMENT:.0f} points. The four measured metrics set the base score; "
            "the regime is context for reading it, not a fifth metric."
        )
    elif regime == "subdued":
        overall = float(np.clip(overall - REGIME_ADJUSTMENT, 0.0, 100.0))
        notes.append(
            f"Current volatility regime is subdued, which nudges the market score down by "
            f"{REGIME_ADJUSTMENT:.0f} points."
        )

    return MarketRiskScore(
        ticker=ticker, volatility=volatility, var_99=var_99, max_drawdown=max_drawdown,
        beta=beta, beta_confidence=beta_confidence, regime=regime,
        volatility_score=volatility_score, var_score=var_score,
        drawdown_score=drawdown_score, beta_score=beta_score,
        overall=overall, excluded=tuple(excluded), notes=tuple(notes),
    )


def score_valuation_risk(valuation: ValuationRisk) -> tuple[float, tuple[str, ...]]:
    """Map a ValuationRisk onto the same 0-100 scale the other two dimensions use.

    Three components: how much of the value sits beyond the forecast (assumption
    dependence), how far a 25bp discount-rate move swings the answer (fragility), and how far
    the model's own WACC sits from what the market price implies (disagreement with the
    market). The third is excluded when Module 2 could not solve for a market-implied rate,
    which happens when the DCF and the price are too far apart for the search range to reach.
    """
    notes: list[str] = []

    dependence_score = _score(valuation.terminal_share, TERMINAL_SHARE_SAFE,
                              TERMINAL_SHARE_DANGEROUS)
    fragility_score = _score(abs(valuation.wacc_sensitivity), WACC_SENSITIVITY_SAFE,
                             WACC_SENSITIVITY_DANGEROUS)

    if valuation.market_implied_wacc is not None:
        gap = abs(valuation.wacc - valuation.market_implied_wacc)
        gap_score = _score(gap, IMPLIED_WACC_GAP_SAFE, IMPLIED_WACC_GAP_DANGEROUS)
    else:
        gap_score = float("nan")
        notes.append(
            "No market-implied discount rate could be solved for, so the valuation-risk "
            "score rests on assumption dependence and fragility only."
        )

    overall = _weighted_average([
        (dependence_score, 0.40), (fragility_score, 0.35), (gap_score, 0.25),
    ])
    return overall, tuple(notes)


@dataclass(frozen=True)
class Divergence:
    """One place the three dimensions disagree, and why it matters."""

    kind: str
    severity: str  # "critical" | "elevated" | "note"
    finding: str

    def render(self) -> str:
        return f"[{self.severity.upper()}] {self.finding}"


@dataclass(frozen=True)
class CompositeRisk:
    """Market, fundamental and valuation risk, on one scale, with the disagreements named."""

    ticker: str
    name: str

    market: MarketRiskScore
    fundamental: FundamentalRisk | None
    valuation: ValuationRisk | None
    valuation_score: float | None

    overall: float
    weights_used: dict[str, float]
    excluded_dimensions: tuple[str, ...]

    divergences: tuple[Divergence, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def rating(self) -> str:
        if self.overall >= 70:
            return "High"
        if self.overall >= 50:
            return "Elevated"
        if self.overall >= 30:
            return "Moderate"
        return "Low"

    @property
    def critical_divergences(self) -> tuple[Divergence, ...]:
        return tuple(d for d in self.divergences if d.severity == "critical")

    def summary(self) -> str:
        dims = ", ".join(f"{k} {v:.0%}" for k, v in self.weights_used.items())
        line = (f"{self.name}: composite risk {self.overall:.0f}/100 ({self.rating}), "
                f"weighted {dims}.")
        if self.excluded_dimensions:
            line += f" Excluded: {', '.join(self.excluded_dimensions)}."
        if self.divergences:
            line += f" {len(self.divergences)} divergence(s) found."
        return line


def _find_divergences(
    market: MarketRiskScore,
    fundamental: FundamentalRisk | None,
    valuation: ValuationRisk | None,
) -> tuple[Divergence, ...]:
    """Where the three dimensions disagree, checked pairwise rather than in the composite."""
    found: list[Divergence] = []

    if fundamental is not None:
        comparison = compare_with_market_risk(fundamental, market.volatility)
        if comparison["verdict"] == "fundamental risk exceeds market risk":
            found.append(Divergence(
                "fundamental_vs_market", "critical",
                f"Fundamental risk ({fundamental.overall_risk:.0f}) exceeds market risk "
                f"({comparison['market_risk']:.0f}): {comparison['reading']}"
            ))
        elif comparison["verdict"] == "market risk exceeds fundamental risk":
            found.append(Divergence(
                "fundamental_vs_market", "note",
                f"Market risk exceeds fundamental risk: {comparison['reading']}"
            ))

    if valuation is not None and valuation.is_fragile and market.regime == "elevated":
        found.append(Divergence(
            "valuation_vs_regime", "elevated",
            f"The valuation moves {abs(valuation.wacc_sensitivity):.1%} for a 25 basis point "
            "change in the discount rate, and the current volatility regime is elevated. The "
            "inputs a sensitive valuation depends on (beta, the discount rate) are least "
            "stable exactly when the valuation is most sensitive to them, which is the worst "
            "time to trust a point estimate from this model."
        ))

    if (fundamental is not None and valuation is not None
            and fundamental.deterioration_risk > 50 and valuation.terminal_share > 0.70):
        found.append(Divergence(
            "deteriorating_vs_terminal", "elevated",
            f"{valuation.terminal_share:.0%} of enterprise value sits in the terminal value, "
            f"which assumes the business continues on roughly its recent trend forever. "
            f"Fundamental deterioration is scoring {fundamental.deterioration_risk:.0f}/100, "
            "so the trend that assumption extrapolates is itself the finding most worth "
            "checking before trusting the valuation."
        ))

    return tuple(found)


def compose(
    ticker: str,
    name: str,
    market: MarketRiskScore,
    fundamental: FundamentalRisk | None = None,
    valuation: ValuationRisk | None = None,
    weights: dict[str, float] | None = None,
) -> CompositeRisk:
    """Combine the three dimensions into one score, with the disagreements between them.

    A dimension that is unavailable (no filings for `fundamental`, no solvable DCF for
    `valuation`) is dropped and the remaining weights renormalised, exactly as Module 1
    renormalises its pillar weights over missing metrics. The result discloses which
    dimensions were used rather than presenting a score that quietly leaned harder on
    whatever happened to be available.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    notes: list[str] = []
    excluded: list[str] = []

    valuation_score = None
    valuation_notes: tuple[str, ...] = ()
    if valuation is not None:
        valuation_score, valuation_notes = score_valuation_risk(valuation)

    components = {"market": market.overall, "fundamental": None, "valuation": valuation_score}
    if fundamental is not None:
        components["fundamental"] = fundamental.overall_risk
    else:
        excluded.append("fundamental")
        notes.append(
            "Fundamental risk is excluded: Module 1 is not installed, or no filed statements "
            "were found for this ticker."
        )
    if valuation is None:
        excluded.append("valuation")
        notes.append(
            "Valuation risk is excluded: Module 2 is not installed, or no DCF could be "
            "solved for this ticker."
        )

    scored = [(components[k], weights[k]) for k in ("market", "fundamental", "valuation")
              if components[k] is not None and not (
                  isinstance(components[k], float) and np.isnan(components[k]))]
    overall = _weighted_average(scored)

    used_weight = sum(w for k, w in weights.items()
                      if components.get(k) is not None
                      and not (isinstance(components[k], float) and np.isnan(components[k])))
    weights_used = ({k: w / used_weight for k, w in weights.items()
                     if components.get(k) is not None
                     and not (isinstance(components[k], float) and np.isnan(components[k]))}
                    if used_weight > 0 else {})

    divergences = _find_divergences(market, fundamental, valuation)
    notes.extend(valuation_notes)
    notes.extend(market.notes)

    return CompositeRisk(
        ticker=ticker, name=name, market=market, fundamental=fundamental,
        valuation=valuation, valuation_score=valuation_score,
        overall=overall, weights_used=weights_used, excluded_dimensions=tuple(excluded),
        divergences=divergences, notes=tuple(notes),
    )


def assess(
    ticker: str,
    period: str = "5y",
    benchmark: str = "^GSPC",
    confidence: float = 0.99,
    weights: dict[str, float] | None = None,
) -> CompositeRisk:
    """Fetch everything live and build the composite view for one ticker.

    Market risk always runs, since it is this module's own data. Fundamental and valuation
    risk degrade gracefully: a missing Module 1 or Module 2 install, a ticker with no filed
    statements, or a DCF that fails to solve all fall back to `None` for that dimension
    rather than raising, because a composite view that stops working the moment one of three
    optional inputs is unavailable is worse than one that says plainly what it could not use.
    """
    from risk_engine.beta import compute_beta
    from risk_engine.data_loader import clean_prices, fetch_prices
    from risk_engine.drawdown import drawdown_summary
    from risk_engine.garch import conditional_var, fit_garch
    from risk_engine.returns import log_returns
    from risk_engine.var_historical import historical_var
    from risk_engine.volatility import historical_volatility, rolling_volatility, volatility_regime

    frame, _ = fetch_prices(ticker, period=period)
    frame = clean_prices(frame)
    returns = log_returns(frame["adj_close"])

    vol = historical_volatility(returns)
    var_result = historical_var(returns, confidence)
    drawdown = drawdown_summary(frame["adj_close"])

    beta_value, beta_confidence = None, None
    try:
        benchmark_frame, _ = fetch_prices(benchmark, period=period)
        benchmark_returns = log_returns(clean_prices(benchmark_frame)["adj_close"])
        beta_result = compute_beta(returns, benchmark_returns)
        beta_value, beta_confidence = beta_result.beta_regression, beta_result.confidence
    except Exception:  # noqa: BLE001 - beta is one of four inputs, not required
        pass

    regime = "normal"
    try:
        rolling = rolling_volatility(returns, window=60).dropna()
        if len(rolling):
            current = float(rolling.iloc[-1])
            regime = volatility_regime(current, float(rolling.mean()))
    except Exception:  # noqa: BLE001
        pass

    market = score_market_risk(
        ticker=ticker, volatility=vol.annualised, var_99=var_result.var_return,
        max_drawdown=drawdown.max_drawdown, beta=beta_value, beta_confidence=beta_confidence,
        regime=regime,
    )

    fundamental_result = None
    try:
        from risk_engine.fundamental import assess as assess_fundamental
        fundamental_result = assess_fundamental(ticker)
    except Exception:  # noqa: BLE001 - degrade to a two-dimension composite
        pass

    valuation_result = None
    try:
        from risk_engine.valuation_risk import assess as assess_valuation
        valuation_result = assess_valuation(ticker)
    except Exception:  # noqa: BLE001
        pass

    name = fundamental_result.name if fundamental_result else ticker

    return compose(ticker, name, market, fundamental_result, valuation_result, weights)
