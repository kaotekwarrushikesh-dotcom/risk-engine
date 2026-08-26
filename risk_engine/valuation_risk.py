"""Phase 10: valuation risk and stress testing, on Module 2's DCF.

Valuation risk is not "the DCF says the share is expensive". That is a *view*, and a view is
not a risk. Valuation risk is **how much the answer moves when an assumption moves**, and the
two are close to independent: a DCF can sit exactly on the market price and still be
worthless, if a quarter-point change in the discount rate moves it by half.

Three things are measured here, and only the first is commonly done.

**Sensitivity.** How far the implied value travels for a defensible change in WACC or terminal
growth. Reported as a grid, because the interaction matters: the two assumptions are not
independent in their effect and a one-at-a-time table hides that.

**Assumption dependence.** The share of enterprise value sitting in the terminal value. A DCF
with 70% of its value beyond the forecast horizon is not a forecast of cash flows, it is a bet
on a perpetuity formula wearing a forecast's clothing. This single number is the most honest
read on how much of a valuation is actually modelled.

**Reverse stress testing.** Rather than asking what the company is worth, ask what would have
to be true for today's price to be right, and then judge whether that is plausible. This
inverts the usual direction and is far harder to fool yourself with, because it produces a
required assumption rather than a comforting answer. Module 2 already solves for the
market-implied discount rate; this module adds the terminal growth the market requires at the
model's own WACC, and measures how far each sits from the model's assumption.

**Everything is computed by Module 2.** This module shocks inputs and re-runs that engine's
own DCF; it does not contain a second discounting implementation. Module 2's documented
calibration finding therefore applies to every number here, and is attached to the result
rather than left in a README: that DCF reads systematically below market, so the *level* is
not the finding. The *sensitivity* is, and sensitivity survives a level bias that moves every
scenario in the same direction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A quarter-point either way on the discount rate is well inside the range two reasonable
# analysts would disagree by, which is what makes it the right size of shock to test.
WACC_SHOCK = 0.0025
GROWTH_SHOCK = 0.0050
# Above this share of enterprise value in the terminal value, the valuation is mostly an
# assumption about perpetuity rather than a forecast of anything.
TERMINAL_DEPENDENCE_HIGH = 0.70
TERMINAL_DEPENDENCE_CRITICAL = 0.85
# A valuation that moves more than this for a 25bp WACC change is fragile to the single input
# analysts most often disagree about.
FRAGILE_PER_25BP = 0.05


@dataclass(frozen=True)
class ValuationRisk:
    """How much a valuation depends on its assumptions, rather than what it concludes."""

    ticker: str
    name: str
    currency: str

    share_price: float
    implied_share_price: float
    upside: float

    wacc: float
    terminal_growth: float
    terminal_share: float

    market_implied_wacc: float | None
    market_implied_growth: float | None

    wacc_sensitivity: float
    growth_sensitivity: float
    grid: pd.DataFrame

    flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def assumption_dependence(self) -> str:
        if self.terminal_share >= TERMINAL_DEPENDENCE_CRITICAL:
            return "Critical"
        if self.terminal_share >= TERMINAL_DEPENDENCE_HIGH:
            return "High"
        if self.terminal_share >= 0.5:
            return "Moderate"
        return "Low"

    @property
    def is_fragile(self) -> bool:
        """Whether a 25bp move in the discount rate materially moves the answer."""
        return abs(self.wacc_sensitivity) > FRAGILE_PER_25BP

    def summary(self) -> str:
        return (
            f"{self.name}: {self.terminal_share:.0%} of enterprise value sits in the terminal "
            f"value ({self.assumption_dependence.lower()} assumption dependence). A 25bp WACC "
            f"change moves the implied price by {abs(self.wacc_sensitivity):.1%}."
        )


def _pipeline(query: str, horizon: int = 5):
    try:
        from valuation_engine import pipeline
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "Valuation risk needs Module 2. Install it with:\n"
            '  pip install "git+https://github.com/kaotekwarrushikesh-dotcom/'
            'valuation-engine.git"\n'
            "The market-risk parts of this engine do not require it."
        ) from exc
    return pipeline.run_quick_pipeline(query, horizon)


def _revalue(base: dict, wacc: float, growth: float) -> float:
    """Re-run Module 2's own DCF with one or both assumptions shocked.

    Returns the implied share price, or NaN where the combination is not solvable. A terminal
    growth at or above the discount rate makes the perpetuity formula divide by zero or go
    negative, which is not a valuation worth reporting and is excluded rather than clamped.
    """
    from valuation_engine.dcf import run_dcf

    if growth >= wacc - 0.005:
        return float("nan")
    try:
        result = run_dcf(
            ticker=base["ticker"], fcff=base["fcff"], wacc=wacc, terminal_growth=growth,
            net_debt=base["net_debt"], shares_outstanding=base["shares"],
            current_share_price=base["share_price"], currency=base["currency"],
            roic=base["roic"],
        )
        return float(result.implied_share_price)
    except Exception:  # noqa: BLE001 - a failed scenario is excluded, not fatal
        return float("nan")


def assess(query: str, horizon: int = 5) -> ValuationRisk:
    """Valuation risk for one company, built on Module 2's DCF."""
    r = _pipeline(query, horizon)

    dcf = r.get("dcf")
    if dcf is None:
        raise ValueError(
            f"Module 2 produced no DCF for {query}: "
            f"{r.get('dcf_error') or 'the WACC failed validation'}. There is no valuation to "
            "stress."
        )

    assumptions = r["assumptions"]
    base = {
        "ticker": r["ticker"], "fcff": r["fcff"], "net_debt": r["net_debt"],
        "shares": float(dcf.shares_outstanding), "share_price": r["share_price"],
        "currency": r["currency"], "roic": assumptions.terminal_roic,
    }

    wacc = float(r["wacc"].wacc)
    growth = float(assumptions.terminal_growth)
    implied = float(dcf.implied_share_price)

    flags: list[str] = []
    notes: list[str] = []

    # --- Sensitivity, measured by actually re-running the model ---------------------------------
    up = _revalue(base, wacc + WACC_SHOCK, growth)
    down = _revalue(base, wacc - WACC_SHOCK, growth)
    wacc_sensitivity = ((down - up) / (2 * implied)) if implied > 0 and not (
        np.isnan(up) or np.isnan(down)) else float("nan")

    g_up = _revalue(base, wacc, growth + GROWTH_SHOCK)
    g_down = _revalue(base, wacc, growth - GROWTH_SHOCK)
    growth_sensitivity = ((g_up - g_down) / (2 * implied)) if implied > 0 and not (
        np.isnan(g_up) or np.isnan(g_down)) else float("nan")

    # --- The grid, because the two assumptions interact -------------------------------------------
    wacc_steps = [wacc + d for d in (-0.010, -0.005, 0.0, 0.005, 0.010)]
    growth_steps = [growth + d for d in (-0.010, -0.005, 0.0, 0.005, 0.010)]
    grid = pd.DataFrame(
        [[_revalue(base, w, g) for g in growth_steps] for w in wacc_steps],
        index=[f"{w:.2%}" for w in wacc_steps],
        columns=[f"{g:.2%}" for g in growth_steps],
    )
    grid.index.name = "WACC"
    grid.columns.name = "Terminal growth"

    # --- Reverse stress test: what must be true for today's price ---------------------------------
    market_implied_wacc = r.get("implied_wacc")
    market_implied_growth = _solve_growth_for_price(base, wacc, r["share_price"])

    # --- Diagnostics -------------------------------------------------------------------------------
    terminal_share = float(dcf.terminal_share)

    if terminal_share >= TERMINAL_DEPENDENCE_CRITICAL:
        flags.append(
            f"{terminal_share:.0%} of enterprise value sits beyond the explicit forecast. At "
            "this level the valuation is a bet on a perpetuity formula rather than a forecast "
            "of cash flows, and the forecast years are close to decorative."
        )
    elif terminal_share >= TERMINAL_DEPENDENCE_HIGH:
        flags.append(
            f"{terminal_share:.0%} of enterprise value is in the terminal value, so most of "
            "the answer rests on assumptions past the horizon rather than on modelled cash."
        )

    if not np.isnan(wacc_sensitivity) and abs(wacc_sensitivity) > FRAGILE_PER_25BP:
        flags.append(
            f"A 25 basis point change in WACC moves the implied price by "
            f"{abs(wacc_sensitivity):.1%}. That is smaller than the range two reasonable "
            "analysts would disagree by, so the precision of this valuation is lower than its "
            "decimal places suggest."
        )

    if market_implied_wacc is not None:
        gap = wacc - float(market_implied_wacc)
        if abs(gap) > 0.02:
            flags.append(
                f"The market price implies a discount rate of {market_implied_wacc:.2%} "
                f"against this model's {wacc:.2%}, a gap of {abs(gap) * 100:.1f} points. "
                "Either the market is pricing a materially different risk profile, or the "
                "model's cost of capital is wrong. The gap is a question rather than a verdict."
            )

    if market_implied_growth is not None and not np.isnan(market_implied_growth):
        if market_implied_growth > 0.05:
            flags.append(
                f"At this model's WACC, the market price requires terminal growth of "
                f"{market_implied_growth:.2%} in perpetuity. Anything above nominal GDP "
                "growth means the company is assumed to grow faster than the economy forever, "
                "which no business has managed."
            )

    notes.append(
        "Module 2's DCF reads systematically below market across its tested universe, a "
        "documented limitation of its terminal-value method rather than a data problem. That "
        "bias moves every scenario here in the same direction, so it affects the level and "
        "not the sensitivity, which is what this phase measures."
    )
    notes.append(
        "Sensitivities are computed by re-running Module 2's own DCF with shocked inputs "
        "rather than by differentiating a formula, so they reflect whatever the engine "
        "actually does, including its cross-checks and floors."
    )

    return ValuationRisk(
        ticker=r["ticker"], name=r["name"], currency=r["currency"],
        share_price=float(r["share_price"]), implied_share_price=implied,
        upside=float(dcf.upside) if hasattr(dcf, "upside") else
              (implied / float(r["share_price"]) - 1.0),
        wacc=wacc, terminal_growth=growth, terminal_share=terminal_share,
        market_implied_wacc=float(market_implied_wacc) if market_implied_wacc else None,
        market_implied_growth=market_implied_growth,
        wacc_sensitivity=wacc_sensitivity, growth_sensitivity=growth_sensitivity,
        grid=grid, flags=tuple(flags), notes=tuple(notes),
    )


def _solve_growth_for_price(base: dict, wacc: float, target_price: float) -> float:
    """The terminal growth rate that makes the DCF agree with the market, at the model's WACC.

    Bisection rather than a closed form, because the target is Module 2's full DCF including
    its reinvestment consistency and cross-checks, not a bare perpetuity. Returns NaN when no
    growth rate inside the searchable range reaches the price, which is itself informative: it
    means the gap cannot be explained by terminal growth alone.
    """
    low, high = -0.05, wacc - 0.006
    if high <= low:
        return float("nan")

    price_low = _revalue(base, wacc, low)
    if np.isnan(price_low):
        return float("nan")

    # The top of the range is where the perpetuity is closest to breaking down, so it is the
    # most likely cell to be unsolvable. Walking `high` down until it prices, rather than
    # giving up, keeps the search alive for the common case where the answer sits comfortably
    # below the boundary and only the extreme edge fails.
    price_high = _revalue(base, wacc, high)
    while np.isnan(price_high) and high > low:
        high -= 0.005
        price_high = _revalue(base, wacc, high)
    if np.isnan(price_high):
        return float("nan")
    if not (price_low <= target_price <= price_high):
        return float("nan")

    for _ in range(60):
        mid = (low + high) / 2
        value = _revalue(base, wacc, mid)
        if np.isnan(value):
            high = mid
            continue
        if value < target_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def stress_test(query: str, scenarios: dict[str, dict] | None = None,
                horizon: int = 5) -> pd.DataFrame:
    """Named stress scenarios, each a full revaluation through Module 2.

    Defaults are deliberately expressed as shocks to the discount rate and terminal outlook
    rather than to revenue, because those are the two inputs this engine can move rigorously
    without rebuilding Module 2's forecast. A revenue or margin shock belongs in Module 2's own
    scenario engine, which recomputes the forecast properly, and duplicating it here with a
    cruder version would be worse than not having it.
    """
    r = _pipeline(query, horizon)
    dcf = r.get("dcf")
    if dcf is None:
        raise ValueError(f"No DCF for {query}, so there is nothing to stress.")

    assumptions = r["assumptions"]
    base = {
        "ticker": r["ticker"], "fcff": r["fcff"], "net_debt": r["net_debt"],
        "shares": float(dcf.shares_outstanding), "share_price": r["share_price"],
        "currency": r["currency"], "roic": assumptions.terminal_roic,
    }
    wacc = float(r["wacc"].wacc)
    growth = float(assumptions.terminal_growth)
    price = float(r["share_price"])

    if scenarios is None:
        scenarios = {
            "Base case": {"wacc": 0.0, "growth": 0.0},
            "Rates rise 100bp": {"wacc": 0.010, "growth": 0.0},
            "Rates rise 200bp": {"wacc": 0.020, "growth": 0.0},
            "Growth stalls (-100bp)": {"wacc": 0.0, "growth": -0.010},
            "Risk premium widens 150bp": {"wacc": 0.015, "growth": 0.0},
            "Stagflation: rates +200bp, growth -100bp": {"wacc": 0.020, "growth": -0.010},
            "Soft landing: rates -100bp, growth +50bp": {"wacc": -0.010, "growth": 0.005},
        }

    rows = []
    base_value = _revalue(base, wacc, growth)
    for name, shock in scenarios.items():
        value = _revalue(base, wacc + shock.get("wacc", 0.0), growth + shock.get("growth", 0.0))
        rows.append({
            "scenario": name,
            "wacc": wacc + shock.get("wacc", 0.0),
            "terminal_growth": growth + shock.get("growth", 0.0),
            "implied_price": value,
            "vs_base": (value / base_value - 1.0) if base_value > 0 and not np.isnan(value)
                       else float("nan"),
            "vs_market": (value / price - 1.0) if price > 0 and not np.isnan(value)
                         else float("nan"),
        })
    return pd.DataFrame(rows)
