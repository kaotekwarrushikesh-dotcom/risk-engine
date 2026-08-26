"""Phase 9: fundamental risk, from Module 1's ratios.

Everything up to here measures **market** risk: how much the share price moves. This phase
measures **fundamental** risk: how fragile the underlying business is. They are different
questions, they are answered from different data, and the interesting cases are the ones
where they disagree.

A share can be quiet for years while leverage builds underneath it, and a volatile share can
sit on an unshakeable balance sheet. Market risk cannot see the first case and overstates the
second, which is why an engine that only measures price movement is incomplete rather than
merely narrow. `compare_with_market_risk()` puts the two side by side and names the
divergence when there is one.

**Nothing here recomputes a ratio.** Every figure comes from Module 1, which already fetches
filings, handles the tag migrations and currency traps, and computes the ratios. Re-deriving
them here would create a second implementation free to drift from the first. This module
reads them and asks a different question of them.

**The question is different from Module 1's, and that is the point.** Module 1 scores
financial *health*: is this a good business. This scores financial *risk*: what could go
wrong. Those diverge in a specific way that matters:

  - **Profitability barely features.** A highly profitable company can still be fragile, and
    profit is what Module 1's score is mostly made of.
  - **Direction outweighs level.** A company at 2.5x interest coverage that was 8x three
    years ago is more dangerous than one that has sat at 2.5x throughout. The first is
    deteriorating and the second is simply a leveraged business model. Module 1's score, which
    reads the latest year, cannot tell them apart. This module measures the slope.
  - **Earnings quality gets its own pillar.** Profit that is not backed by cash is the single
    most useful early warning in published accounts, and it is invisible in a margin.

**Wording is deliberately careful.** A ratio cannot prove misconduct, and this module does not
claim it does. A persistent gap between profit and cash is described as an earnings-quality
question worth investigating, because that is what the evidence supports.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Interest coverage below this is where a downturn starts threatening the ability to service
# debt at all, rather than merely being uncomfortable.
COVERAGE_CRITICAL = 2.0
COVERAGE_WARNING = 4.0
# Current ratio below 1 means short-term obligations exceed short-term assets.
LIQUIDITY_CRITICAL = 0.8
LIQUIDITY_WARNING = 1.0
# Cash conversion: profit not backed by cash over a sustained period.
CONVERSION_WARNING = 0.80
CONVERSION_CRITICAL = 0.60
# Debt-to-equity above this is a materially leveraged balance sheet for a non-financial.
LEVERAGE_WARNING = 1.5
LEVERAGE_CRITICAL = 3.0
# A trend this steep over the observed period is deterioration rather than noise.
DETERIORATION_YEARS = 4


@dataclass(frozen=True)
class RiskFlag:
    """One identified fundamental risk, with the evidence behind it."""

    category: str
    severity: str  # "critical" | "elevated" | "watch"
    finding: str
    evidence: str
    implication: str

    def render(self) -> str:
        return f"[{self.severity.upper()}] {self.category}: {self.finding} ({self.evidence})"


@dataclass(frozen=True)
class FundamentalRisk:
    """A business's fragility, as distinct from its share price's volatility."""

    ticker: str
    name: str
    currency: str
    source: str
    years: int
    first_year: int
    last_year: int

    leverage_risk: float
    liquidity_risk: float
    earnings_quality_risk: float
    deterioration_risk: float
    overall_risk: float

    latest: dict
    trends: dict
    flags: tuple[RiskFlag, ...]
    health_score: float | None
    health_rating: str | None
    unavailable: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def rating(self) -> str:
        """Plain-language band. Higher score means more risk, the inverse of Module 1."""
        if self.overall_risk >= 70:
            return "High"
        if self.overall_risk >= 50:
            return "Elevated"
        if self.overall_risk >= 30:
            return "Moderate"
        return "Low"

    @property
    def critical_flags(self) -> tuple[RiskFlag, ...]:
        return tuple(f for f in self.flags if f.severity == "critical")

    def summary(self) -> str:
        return (
            f"{self.name}: fundamental risk {self.overall_risk:.0f}/100 ({self.rating}). "
            f"Leverage {self.leverage_risk:.0f}, liquidity {self.liquidity_risk:.0f}, "
            f"earnings quality {self.earnings_quality_risk:.0f}, deterioration "
            f"{self.deterioration_risk:.0f}. {len(self.flags)} flag(s), "
            f"{len(self.critical_flags)} critical."
        )


def _score(value: float, safe: float, dangerous: float) -> float:
    """Map a ratio onto a 0-100 risk score, where 100 is the most risk.

    `safe` scores 0 and `dangerous` scores 100, with linear interpolation between and clamping
    outside. Either ordering works, so a metric where lower is riskier simply has `safe` above
    `dangerous`.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    span = dangerous - safe
    if span == 0:
        return float("nan")
    return float(np.clip((value - safe) / span, 0.0, 1.0) * 100.0)


def _slope_per_year(series: pd.Series) -> float:
    """Least-squares slope of a ratio against time, in units per year.

    A slope rather than a start-to-end difference, because a single unusual endpoint would
    otherwise decide whether a company is called deteriorating. Regression uses every year.
    """
    clean = series.dropna()
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    return float(np.polyfit(x, clean.to_numpy(dtype=float), 1)[0])


def _mean_available(values: list[float]) -> float:
    usable = [v for v in values if not (v is None or np.isnan(v))]
    return float(np.mean(usable)) if usable else float("nan")


def assess(query: str, years: int = DETERIORATION_YEARS) -> FundamentalRisk:
    """Fundamental risk for one company, computed from Module 1's ratios.

    `years` sets how much recent history the deterioration slopes are measured over. Older
    years still inform the levels but a five-year-old trend is not what makes a company risky
    today.
    """
    try:
        from fsi.company import analyse
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "Fundamental risk needs Module 1. Install it with:\n"
            '  pip install "git+https://github.com/kaotekwarrushikesh-dotcom/'
            'financial-statement-intelligence.git"\n'
            "The market-risk parts of this engine do not require it."
        ) from exc

    analysis = analyse(query)
    ratios = analysis.ratios
    latest = ratios.iloc[-1]
    recent = ratios.tail(max(years, 3))

    flags: list[RiskFlag] = []
    notes: list[str] = []
    unavailable: list[str] = []

    def value(column: str) -> float:
        raw = latest.get(column, float("nan"))
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            unavailable.append(column)
            return float("nan")
        return float(raw)

    coverage = value("interest_coverage")
    debt_equity = value("debt_to_equity")
    current = value("current_ratio")
    conversion = value("cfo_to_net_income")
    fcf_margin = value("fcf_margin")

    # --- Leverage ----------------------------------------------------------------------------
    leverage_risk = _mean_available([
        _score(debt_equity, 0.3, LEVERAGE_CRITICAL),
        _score(coverage, 15.0, COVERAGE_CRITICAL),
    ])

    if not np.isnan(coverage) and coverage < COVERAGE_CRITICAL:
        flags.append(RiskFlag(
            "Leverage", "critical",
            "Interest cover is thin enough that a downturn threatens debt service",
            f"EBIT covers interest {coverage:.1f}x in FY{int(latest['fiscal_year'])}",
            "A moderate fall in operating profit would leave the company unable to service "
            "its debt from earnings."))
    elif not np.isnan(coverage) and coverage < COVERAGE_WARNING:
        flags.append(RiskFlag(
            "Leverage", "elevated", "Interest cover leaves limited headroom",
            f"EBIT covers interest {coverage:.1f}x",
            "Manageable now, but it removes the cushion for a bad year."))

    if not np.isnan(debt_equity) and debt_equity > LEVERAGE_CRITICAL:
        flags.append(RiskFlag(
            "Leverage", "critical", "Debt is large relative to the equity base",
            f"Debt-to-equity {debt_equity:.2f}",
            "Losses erode a thin equity cushion quickly, and refinancing terms are more "
            "sensitive to conditions."))

    # --- Liquidity ---------------------------------------------------------------------------
    liquidity_risk = _score(current, 2.5, LIQUIDITY_CRITICAL)

    if not np.isnan(current) and current < LIQUIDITY_CRITICAL:
        flags.append(RiskFlag(
            "Liquidity", "critical", "Short-term obligations exceed short-term assets",
            f"Current ratio {current:.2f}",
            "The company depends on continued access to funding or on operating cash "
            "arriving on schedule."))
    elif not np.isnan(current) and current < LIQUIDITY_WARNING:
        flags.append(RiskFlag(
            "Liquidity", "watch", "Working capital is tight",
            f"Current ratio {current:.2f}",
            "Common in businesses that collect cash before paying suppliers, so read it "
            "alongside cash conversion rather than alone."))

    # --- Earnings quality --------------------------------------------------------------------
    conversion_series = recent["cfo_to_net_income"].dropna()
    sustained = (float(conversion_series.mean()) if len(conversion_series) >= 3
                 else float("nan"))
    earnings_quality_risk = _mean_available([
        _score(sustained, 1.3, CONVERSION_CRITICAL),
        _score(fcf_margin, 0.20, -0.05),
    ])

    if not np.isnan(sustained) and sustained < CONVERSION_CRITICAL:
        flags.append(RiskFlag(
            "Earnings quality", "critical",
            "Reported profit is persistently not backed by operating cash",
            f"Operating cash flow averages {sustained:.2f}x net income over "
            f"{len(conversion_series)} years",
            "An earnings-quality question worth investigating in the accounting policies "
            "rather than a conclusion about them. Sustained divergence usually reflects "
            "revenue recognised before cash arrives."))
    elif not np.isnan(sustained) and sustained < CONVERSION_WARNING:
        flags.append(RiskFlag(
            "Earnings quality", "elevated",
            "Operating cash flow runs below reported profit",
            f"Averages {sustained:.2f}x net income",
            "Worth reading alongside working capital movements before drawing a conclusion."))

    if not np.isnan(fcf_margin) and fcf_margin < 0:
        flags.append(RiskFlag(
            "Cash generation", "elevated", "Free cash flow is negative",
            f"Free cash flow margin {fcf_margin:.1%}",
            "The business consumes cash after capital spending, so it depends on existing "
            "reserves or external funding to continue at this rate."))

    # --- Deterioration: direction rather than level ---------------------------------------------
    trends = {
        "interest_coverage": _slope_per_year(recent["interest_coverage"]),
        "current_ratio": _slope_per_year(recent["current_ratio"]),
        "debt_to_equity": _slope_per_year(recent["debt_to_equity"]),
        "ebitda_margin": _slope_per_year(recent["ebitda_margin"]),
        "cfo_to_net_income": _slope_per_year(recent["cfo_to_net_income"]),
    }

    # Each slope is scored against a per-year move large enough to matter over a few years.
    deterioration_risk = _mean_available([
        _score(trends["interest_coverage"], 1.0, -2.0),
        _score(trends["current_ratio"], 0.1, -0.2),
        _score(trends["debt_to_equity"], -0.1, 0.4),
        _score(trends["ebitda_margin"], 0.01, -0.03),
        _score(trends["cfo_to_net_income"], 0.05, -0.15),
    ])

    if not np.isnan(trends["interest_coverage"]) and trends["interest_coverage"] < -1.0:
        flags.append(RiskFlag(
            "Deterioration", "elevated", "Interest cover is falling year on year",
            f"Down {abs(trends['interest_coverage']):.1f}x per year over the last "
            f"{len(recent)} years",
            "The level may still look adequate, but the direction is what turns an adequate "
            "balance sheet into a stressed one."))

    if not np.isnan(trends["ebitda_margin"]) and trends["ebitda_margin"] < -0.02:
        flags.append(RiskFlag(
            "Deterioration", "elevated", "Operating margin is compressing",
            f"EBITDA margin down {abs(trends['ebitda_margin']) * 100:.1f} points per year",
            "Sustained compression reduces the cushion absorbing a demand or cost shock."))

    if not np.isnan(trends["debt_to_equity"]) and trends["debt_to_equity"] > 0.3:
        flags.append(RiskFlag(
            "Deterioration", "elevated", "Leverage is rising",
            f"Debt-to-equity up {trends['debt_to_equity']:.2f} per year",
            "Rising leverage into a stable business is a choice; into a weakening one it "
            "compounds the existing risk."))

    overall = _mean_available([
        leverage_risk, liquidity_risk, earnings_quality_risk, deterioration_risk])

    if unavailable:
        notes.append(
            f"{len(set(unavailable))} metric(s) unavailable and excluded rather than scored "
            f"as safe: {', '.join(sorted(set(unavailable)))}. A missing interest expense in "
            "particular means the company stopped tagging it, not that borrowing is free."
        )
    notes.append(
        f"Levels are read from FY{int(latest['fiscal_year'])}; deterioration slopes are "
        f"regressions over the last {len(recent)} years, so a single unusual year cannot "
        "decide whether a company is called deteriorating."
    )
    notes.append(
        "Every ratio here comes from Module 1 rather than being recomputed, so there is one "
        "implementation of each and no second copy free to drift from it."
    )

    return FundamentalRisk(
        ticker=analysis.ticker, name=analysis.name, currency=analysis.currency,
        source=analysis.source, years=analysis.years,
        first_year=analysis.first_year, last_year=analysis.last_year,
        leverage_risk=leverage_risk, liquidity_risk=liquidity_risk,
        earnings_quality_risk=earnings_quality_risk, deterioration_risk=deterioration_risk,
        overall_risk=overall,
        latest={
            "interest_coverage": coverage, "debt_to_equity": debt_equity,
            "current_ratio": current, "cfo_to_net_income": conversion,
            "fcf_margin": fcf_margin,
            "ebitda_margin": float(latest.get("ebitda_margin", float("nan"))),
        },
        trends=trends,
        flags=tuple(flags),
        health_score=analysis.health.get("overall"),
        health_rating=analysis.health.get("rating"),
        unavailable=tuple(sorted(set(unavailable))),
        notes=tuple(notes),
    )


def compare_with_market_risk(
    fundamental: FundamentalRisk,
    annualised_volatility: float,
    peer_volatility: float | None = None,
) -> dict:
    """Put fragility and price movement side by side, and name the divergence.

    The cases worth catching are the two where the measures disagree. A quiet share over a
    weakening balance sheet is the more dangerous of the two, because the market's calm is
    doing the work of reassurance that the accounts do not support. The reverse, a volatile
    share over a solid business, is where price risk overstates business risk.

    `peer_volatility` supplies a comparison point. Without one the volatility bands fall back
    to broad equity-market norms, which are a cruder instrument and are labelled as such.
    """
    if peer_volatility and peer_volatility > 0:
        market_risk = float(np.clip(annualised_volatility / peer_volatility, 0, 2) / 2 * 100)
        basis = f"relative to a peer volatility of {peer_volatility:.1%}"
    else:
        # 15% annualised is unremarkable for a large listed equity, 55% is very high.
        market_risk = _score(annualised_volatility, 0.15, 0.55)
        basis = "against broad equity-market norms, which is a cruder comparison than peers"

    gap = fundamental.overall_risk - market_risk

    if gap > 25:
        verdict = "fundamental risk exceeds market risk"
        reading = (
            f"The business looks more fragile than the share price suggests. Fundamental risk "
            f"is {fundamental.overall_risk:.0f} against a market-risk read of "
            f"{market_risk:.0f}. This is the more dangerous direction of the two: a calm share "
            "price is not evidence about the balance sheet, and volatility-based measures "
            "cannot see leverage building underneath a quiet stock."
        )
    elif gap < -25:
        verdict = "market risk exceeds fundamental risk"
        reading = (
            f"The share price moves more than the accounts justify. Market risk reads "
            f"{market_risk:.0f} against fundamental risk of {fundamental.overall_risk:.0f}. "
            "Price volatility is overstating business fragility here, which is the safer "
            "direction to be wrong in but still a mispricing of risk."
        )
    else:
        verdict = "the two measures broadly agree"
        reading = (
            f"Fundamental risk ({fundamental.overall_risk:.0f}) and market risk "
            f"({market_risk:.0f}) point the same way, so the share price appears to be "
            "reflecting the state of the business rather than diverging from it."
        )

    return {
        "fundamental_risk": fundamental.overall_risk,
        "market_risk": market_risk,
        "gap": gap,
        "verdict": verdict,
        "reading": reading,
        "basis": basis,
    }
