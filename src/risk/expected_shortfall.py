"""Phase 7: Expected Shortfall, and a comparison across every VaR method built so far.

    ES(c) = E[loss | loss > VaR(c)]

Expected Shortfall answers the question VaR structurally cannot: *given that the bad case
happens, how bad is it on average?* VaR is a threshold and says nothing whatever about what
lies beyond it. Two portfolios can share an identical 99% VaR while one loses a further 1%
in the tail and the other loses everything, and no VaR number at any confidence level
distinguishes them. ES does, because it averages the losses past the threshold instead of
reporting where the threshold sits.

**ES is coherent; VaR is not.** A risk measure is called coherent if, among other properties,
it is *subadditive*: combining two positions can never produce more risk than holding them
apart, because diversification cannot hurt. VaR violates this. It is possible to construct
two portfolios where VaR(A + B) exceeds VaR(A) + VaR(B), which says diversification increased
risk, and that is a property of the measure rather than of the portfolios. ES satisfies
subadditivity by construction. This is not academic hair-splitting: it is the reason the
Basel Committee moved market-risk capital from 99% VaR to 97.5% ES in the Fundamental Review
of the Trading Book. `demonstrate_var_subadditivity_failure()` builds the counterexample
explicitly rather than asserting the claim.

**Every method gets its own ES, computed the same way it computed its VaR**, so the
comparison is like for like: the historical ES averages the observed losses past the
empirical quantile, the parametric ES uses the closed form for its assumed distribution, and
the Monte Carlo ES averages the simulated paths past the simulated quantile.

**The closed forms are used where they exist rather than approximated.** For a normal
distribution, `ES = mu + sigma * phi(z) / (1 - c)` where phi is the standard normal density
at the quantile. Averaging simulated draws would give the same answer more slowly and with
noise attached, so the analytic result is used and a test checks it against a large
simulation. The Student-t has its own closed form with the same `sqrt((v-2)/v)` unit-variance
rescaling Phase 5 needs, and getting that wrong inflates ES exactly as it inflates VaR.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.risk.var_historical import historical_var
from src.risk.var_monte_carlo import monte_carlo_var, simulate_paths
from src.risk.var_parametric import estimate_degrees_of_freedom, fit_distribution, parametric_var

MIN_TAIL_OBSERVATIONS = 5


@dataclass(frozen=True)
class ExpectedShortfallResult:
    """One ES estimate, alongside the VaR threshold it sits behind."""

    method: str
    confidence: float
    horizon_days: int
    var_return: float
    es_return: float
    es_value: float | None
    tail_observations: float
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def es_pct(self) -> float:
        return self.es_return * 100.0

    @property
    def tail_severity(self) -> float:
        """How much worse the average tail loss is than the threshold, as a ratio.

        A value near 1 means losses beyond VaR are barely worse than VaR itself; a large one
        means the tail keeps going well past the threshold, which is precisely the risk a
        VaR figure on its own conceals.
        """
        return self.es_return / self.var_return if self.var_return > 0 else float("nan")

    def summary(self) -> str:
        horizon = "1 day" if self.horizon_days == 1 else f"{self.horizon_days} days"
        line = (f"{self.confidence:.0%} {horizon} {self.method} ES: {self.es_pct:.2f}% "
                f"(VaR {self.var_return * 100:.2f}%, {self.tail_severity:.2f}x)")
        if self.es_value is not None:
            line += f" ({self.es_value:,.0f})"
        return line


def _to_simple_loss(log_return: float) -> float:
    return float(-(np.exp(log_return) - 1.0))


def tail_observations_for(n: int, confidence: float) -> int:
    """How many of the worst observations make up the (1 - c) tail.

    Rounded before the ceiling, which is not fussiness: `1 - 0.95` is 0.050000000000000044 in
    binary floating point, so `ceil(1000 * (1 - 0.95))` is 51 rather than 50. That single
    extra observation is the least severe one in the tail, so it drags every ES slightly
    toward the middle of the distribution, in the direction that understates risk, and it
    would never look wrong on inspection.

    At least one, so an ES is always an average of something rather than of nothing.
    """
    return max(1, int(np.ceil(round(n * (1.0 - confidence), 9))))


def empirical_tail(values: np.ndarray, confidence: float) -> np.ndarray:
    """The worst `ceil(n * (1 - c))` observations.

    Defined by rank rather than as `values <= quantile`, which looks equivalent and is not.
    When the quantile lands on a repeated value (a distribution with an atom, or simply a
    tie), the comparison selects every observation sharing that value and the "tail" can
    silently swallow the entire sample: a 5% ES then reports the mean of everything, which is
    not a tail statistic at all and is not obviously wrong on inspection. Taking a fixed
    number of the worst observations cannot do that.
    """
    k = tail_observations_for(len(values), confidence)
    return np.sort(values)[:k]


def historical_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
) -> ExpectedShortfallResult:
    """The average of observed losses beyond the empirical quantile.

    Inherits historical VaR's ceiling and then some: ES is an average *of the tail*, so it
    rests on even fewer observations than the VaR threshold does. At 99% on 250 days the
    threshold has about 2.5 points behind it and the average past it has about 2.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")

    clean = returns.dropna()
    values = clean.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two return observations are needed")

    var_result = historical_var(clean, confidence, horizon_days, portfolio_value)

    scale = np.sqrt(horizon_days)
    tail = empirical_tail(values, confidence)
    warnings: list[str] = []

    if len(tail) == 0:
        return ExpectedShortfallResult(
            method="historical", confidence=confidence, horizon_days=horizon_days,
            var_return=var_result.var_return, es_return=float("nan"), es_value=None,
            tail_observations=0,
            warnings=("no observations fall beyond the quantile, so there is no tail to average",),
        )

    mean_tail_log = float(np.mean(tail)) * scale
    es_return = _to_simple_loss(mean_tail_log)

    if len(tail) < MIN_TAIL_OBSERVATIONS:
        warnings.append(
            f"the ES averages only {len(tail)} observations. An average of a handful of points "
            "is a description of those points rather than an estimate of the tail."
        )

    notes = [
        f"Mean of the {len(tail)} observed losses beyond the {confidence:.0%} quantile.",
    ]
    if horizon_days > 1:
        notes.append(
            f"Scaled to {horizon_days} days by sqrt({horizon_days}), assuming independence."
        )

    return ExpectedShortfallResult(
        method="historical", confidence=confidence, horizon_days=horizon_days,
        var_return=var_result.var_return, es_return=es_return,
        es_value=es_return * portfolio_value if portfolio_value is not None else None,
        tail_observations=float(len(tail)),
        warnings=tuple(warnings), notes=tuple(notes),
    )


def parametric_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    distribution: str = "normal",
) -> ExpectedShortfallResult:
    """ES from the closed form of the assumed distribution.

    Normal:     ES = mu - sigma * phi(z) / (1 - c)
    Student-t:  ES = mu - sigma * s * (v + z^2) / (v - 1) * f_v(z) / (1 - c)

    where `s = sqrt((v - 2) / v)` rescales the t to unit variance, the same correction Phase
    5 applies to VaR and for the same reason.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")
    if distribution not in ("normal", "t"):
        raise ValueError(f"unknown distribution: {distribution!r}, expected 'normal' or 't'")

    fit = fit_distribution(returns)
    var_result = parametric_var(returns, confidence, horizon_days, portfolio_value, distribution)

    tail = 1.0 - confidence
    mu = fit.mean * horizon_days
    sigma = fit.std * np.sqrt(horizon_days)
    warnings = list(var_result.warnings)
    notes: list[str] = []

    if distribution == "normal":
        z = float(stats.norm.ppf(tail))
        es_quantile = mu - sigma * float(stats.norm.pdf(z)) / tail
        notes.append(
            f"Closed form: mu - sigma * phi({z:.3f}) / {tail:.2f}. Used rather than averaging "
            "simulated draws, which would give the same answer more slowly and with noise."
        )
    else:
        v = estimate_degrees_of_freedom(returns)
        z = float(stats.t.ppf(tail, v))
        unit = np.sqrt((v - 2.0) / v)
        # Standard closed form for the t, then rescaled to unit variance so sigma means what
        # it says. Omitting `unit` inflates ES exactly as it inflates VaR.
        es_standard = -(v + z**2) / (v - 1.0) * float(stats.t.pdf(z, v)) / tail
        es_quantile = mu + sigma * unit * es_standard
        notes.append(
            f"Closed form for a Student-t with {v:.1f} degrees of freedom, rescaled by "
            f"sqrt((v-2)/v) = {unit:.3f} to unit variance."
        )

    es_return = _to_simple_loss(es_quantile)

    return ExpectedShortfallResult(
        method=f"parametric-{distribution}", confidence=confidence, horizon_days=horizon_days,
        var_return=var_result.var_return, es_return=es_return,
        es_value=es_return * portfolio_value if portfolio_value is not None else None,
        tail_observations=float("nan"),
        warnings=tuple(warnings), notes=tuple(notes),
    )


def monte_carlo_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    paths: int = 10_000,
    draw_method: str = "bootstrap",
    seed: int = 20250820,
) -> ExpectedShortfallResult:
    """ES as the average of simulated paths beyond the simulated quantile."""
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")

    clean = returns.dropna()
    var_result = monte_carlo_var(clean, confidence, horizon_days, portfolio_value,
                                 paths, draw_method, seed)

    dof = estimate_degrees_of_freedom(clean) if draw_method == "t" else None
    rng = np.random.default_rng(seed)
    simulated = simulate_paths(clean, paths, horizon_days, draw_method, rng, dof)

    tail = empirical_tail(simulated, confidence)
    es_return = _to_simple_loss(float(np.mean(tail)))

    return ExpectedShortfallResult(
        method=f"monte-carlo-{draw_method}", confidence=confidence, horizon_days=horizon_days,
        var_return=var_result.var_return, es_return=es_return,
        es_value=es_return * portfolio_value if portfolio_value is not None else None,
        tail_observations=float(len(tail)),
        warnings=var_result.warnings,
        notes=(f"Mean of {len(tail):,} simulated paths beyond the {confidence:.0%} quantile, "
               f"drawn by {draw_method}.",),
    )


def compare_methods(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    paths: int = 20_000,
) -> pd.DataFrame:
    """Every VaR and ES method side by side, which is the point of having built them all.

    A single VaR number is an opinion dressed as a measurement. Five of them, with their
    assumptions visible and their disagreements on the same row, show which assumption is
    doing the work: if the normal figure sits well below the historical one, the gap is the
    fat tail; if the bootstrap and historical agree, resampling has not distorted anything;
    if the Student-t exceeds every other, the fitted tail is doing something the sample never
    showed.
    """
    rows = []

    hist_es = historical_expected_shortfall(returns, confidence, horizon_days, portfolio_value)
    rows.append({
        "method": "Historical", "assumption": "none (empirical quantile)",
        "var": hist_es.var_return, "es": hist_es.es_return,
        "tail_severity": hist_es.tail_severity,
    })

    for dist, label in (("normal", "Parametric (normal)"), ("t", "Parametric (Student-t)")):
        es = parametric_expected_shortfall(returns, confidence, horizon_days,
                                           portfolio_value, dist)
        rows.append({
            "method": label,
            "assumption": "returns are normal" if dist == "normal"
                          else "returns are Student-t, v fitted",
            "var": es.var_return, "es": es.es_return, "tail_severity": es.tail_severity,
        })

    for draw, label in (("bootstrap", "Monte Carlo (bootstrap)"), ("t", "Monte Carlo (t)")):
        es = monte_carlo_expected_shortfall(returns, confidence, horizon_days,
                                            portfolio_value, paths, draw)
        rows.append({
            "method": label,
            "assumption": "resampled from observed returns" if draw == "bootstrap"
                          else "simulated from a fitted Student-t",
            "var": es.var_return, "es": es.es_return, "tail_severity": es.tail_severity,
        })

    frame = pd.DataFrame(rows)
    frame["var_vs_historical"] = frame["var"] - frame["var"].iloc[0]
    return frame


def demonstrate_var_subadditivity_failure() -> dict:
    """A concrete counterexample where VaR says diversification increased risk.

    Two independent positions, each of which loses heavily with probability 4%. At 95%
    confidence, neither position's own 5% tail reaches its loss, so each has a VaR of 0. The
    combined portfolio loses on either event, so its probability of loss is roughly 8%, which
    *does* reach into the 95% tail, and the combined VaR is therefore positive and larger than
    the sum of the parts.

    Diversification cannot increase risk. VaR says it did here, which is a defect of the
    measure rather than a fact about the positions. ES, being an average over the whole tail
    rather than a single threshold, does not have this failure mode, and this function checks
    that too rather than asserting it.
    """
    n = 100_000
    rng = np.random.default_rng(1234)

    # Each position: a 4% chance of a large loss, otherwise a small gain.
    a = np.where(rng.random(n) < 0.04, -1.0, 0.02)
    b = np.where(rng.random(n) < 0.04, -1.0, 0.02)
    combined = (a + b) / 2.0  # an equally weighted portfolio of the two

    def var_95(x):
        return -float(np.quantile(x, 0.05))

    def es_95(x):
        return -float(np.mean(empirical_tail(x, 0.95)))

    var_a, var_b, var_ab = var_95(a), var_95(b), var_95(combined)
    es_a, es_b, es_ab = es_95(a), es_95(b), es_95(combined)

    return {
        "var_a": var_a, "var_b": var_b, "var_combined": var_ab,
        "var_sum_of_parts": var_a + var_b,
        "var_is_subadditive": var_ab <= var_a + var_b + 1e-12,
        "es_a": es_a, "es_b": es_b, "es_combined": es_ab,
        "es_sum_of_parts": es_a + es_b,
        "es_is_subadditive": es_ab <= es_a + es_b + 1e-12,
        "reading": (
            f"VaR(A) = {var_a:.2f}, VaR(B) = {var_b:.2f}, but VaR(A+B) = {var_ab:.2f}, which "
            f"exceeds their sum of {var_a + var_b:.2f}. VaR says combining two independent "
            "positions created risk. ES, by contrast, gives "
            f"ES(A+B) = {es_ab:.2f} against a sum of {es_a + es_b:.2f}, and stays subadditive. "
            "This is why Basel moved market-risk capital from 99% VaR to 97.5% ES."
        ),
    }
