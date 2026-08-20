"""Phase 12: VaR backtesting. Kupiec, Christoffersen, and the traffic light.

Pulled forward from its roadmap position because it is what decides whether Phase 8's
conditional VaR is genuinely better than the unconditional VaR of Phases 4 to 7, rather than
merely more sophisticated. Without a backtest, "GARCH is the right model" is an appeal to
authority; with one, it is a measurement.

**A VaR model is judged on two independent properties, and passing one proves nothing about
the other.**

  Unconditional coverage   Are there about the right *number* of breaches? A 95% VaR should
  (Kupiec, 1995)           be exceeded on about 5% of days. Too many and the model
                           understates risk; too few and it overstates it, which is a
                           calibration failure too rather than a safe cushion, since capital
                           held against a phantom loss is capital not doing anything.

  Independence             Are the breaches *spread out*, or do they arrive in clusters? A
  (Christoffersen, 1998)   model can produce exactly 5% breaches and still be badly wrong if
                           all of them land in the same fortnight, because it means the model
                           never adapted to a stress period; it was simply too high the rest
                           of the time to compensate. This is the test an unconditional VaR
                           is *expected* to fail, and the reason it exists.

Both are likelihood-ratio tests against a chi-squared distribution, and the conditional
coverage test is their sum, which is jointly distributed chi-squared with two degrees of
freedom. Reporting the joint test alone would hide which property failed, so all three are
reported.

**The traffic light is the Basel supervisory framework**, included because it is what a
regulator actually applies to a trading book: on 250 days at 99%, up to 4 breaches is green,
5 to 9 is yellow (a capital multiplier is applied), and 10 or more is red. It is a cruder
instrument than the likelihood-ratio tests and it is the one with consequences attached.

**A test that does not reject is not a test that confirms.** With 250 observations these
tests have low power: a genuinely mediocre model routinely survives them. Every result here
carries its sample size for that reason, and "passed" means "not rejected at this sample
size", which is a weaker statement than it looks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Basel traffic-light zones, defined for 250 trading days at 99% confidence.
BASEL_WINDOW = 250
BASEL_CONFIDENCE = 0.99
BASEL_GREEN_MAX = 4
BASEL_YELLOW_MAX = 9
# Below this many observations the likelihood-ratio tests have too little power to mean much.
LOW_POWER_OBSERVATIONS = 500
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class BacktestResult:
    """The three tests, plus what the numbers mean and how much to trust them."""

    n_observations: int
    n_breaches: int
    expected_breaches: float
    observed_rate: float
    expected_rate: float

    kupiec_statistic: float
    kupiec_p: float
    christoffersen_statistic: float
    christoffersen_p: float
    conditional_coverage_statistic: float
    conditional_coverage_p: float

    max_consecutive_breaches: int
    alpha: float = DEFAULT_ALPHA
    warnings: tuple[str, ...] = ()

    @property
    def passes_coverage(self) -> bool:
        """Right number of breaches."""
        return self.kupiec_p >= self.alpha

    @property
    def passes_independence(self) -> bool:
        """Breaches not clustered."""
        return self.christoffersen_p >= self.alpha

    @property
    def passes_joint(self) -> bool:
        return self.conditional_coverage_p >= self.alpha

    @property
    def verdict(self) -> str:
        if self.passes_coverage and self.passes_independence:
            return "not rejected on either property"
        if not self.passes_coverage and not self.passes_independence:
            return "rejected on both count and clustering"
        if not self.passes_coverage:
            direction = "too many" if self.observed_rate > self.expected_rate else "too few"
            return f"rejected on count ({direction} breaches)"
        return "rejected on clustering: the right number of breaches, arriving together"

    def reading(self) -> str:
        return (
            f"{self.n_breaches} breaches in {self.n_observations} days "
            f"({self.observed_rate:.2%} against an expected {self.expected_rate:.2%}); "
            f"Kupiec p = {self.kupiec_p:.3f}, Christoffersen p = {self.christoffersen_p:.3f}. "
            f"Model {self.verdict}."
        )


def _breach_series(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """1 where the realised loss exceeded that day's VaR, 0 otherwise."""
    aligned = pd.DataFrame({"return": returns, "var": var_series}).dropna()
    realised_loss = -(np.exp(aligned["return"]) - 1.0)
    return (realised_loss > aligned["var"]).astype(int)


def kupiec_test(n: int, breaches: int, expected_rate: float) -> tuple[float, float]:
    """Unconditional coverage: is the breach *count* consistent with the confidence level?

    Likelihood ratio of the observed breach rate against the expected one, distributed
    chi-squared with one degree of freedom. Rejects in both directions: a model breached far
    less than its level claims is miscalibrated too, not merely cautious.
    """
    if n == 0:
        return float("nan"), float("nan")
    observed_rate = breaches / n

    # Both boundary cases have a defined limit; computing them directly would take log(0).
    if breaches == 0:
        statistic = -2.0 * n * np.log(1 - expected_rate)
    elif breaches == n:
        statistic = -2.0 * n * np.log(expected_rate)
    else:
        log_null = (breaches * np.log(expected_rate)
                    + (n - breaches) * np.log(1 - expected_rate))
        log_alt = (breaches * np.log(observed_rate)
                   + (n - breaches) * np.log(1 - observed_rate))
        statistic = -2.0 * (log_null - log_alt)

    return float(statistic), float(stats.chi2.sf(statistic, 1))


def christoffersen_independence_test(breaches: pd.Series) -> tuple[float, float]:
    """Independence: do breaches cluster?

    Fits a first-order Markov chain to the breach sequence and tests whether the probability
    of a breach depends on whether yesterday was one. Under a well-specified model it should
    not: a breach tomorrow should be no more likely because one happened today.

    This is the test an unconditional VaR is expected to fail. A fixed threshold cannot rise
    during a stress period, so its breaches bunch into exactly the weeks the model failed to
    anticipate, even when the total count over a decade looks correct.
    """
    values = breaches.to_numpy()
    if len(values) < 2:
        return float("nan"), float("nan")

    # Transition counts: n_ij = moves from state i to state j.
    previous, current = values[:-1], values[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    # With no breaches, or none following a breach, the alternative model is unidentified and
    # there is no clustering to detect; not rejecting is the correct answer rather than an error.
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0

    pi_01 = n01 / (n00 + n01)   # breach tomorrow given no breach today
    pi_11 = n11 / (n10 + n11)   # breach tomorrow given a breach today
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi in (0.0, 1.0):
        return 0.0, 1.0

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    log_null = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    log_alt = (n00 * safe_log(1 - pi_01) + n01 * safe_log(pi_01)
               + n10 * safe_log(1 - pi_11) + n11 * safe_log(pi_11))

    statistic = -2.0 * (log_null - log_alt)
    statistic = max(statistic, 0.0)  # numerical floor; the LR cannot be negative
    return float(statistic), float(stats.chi2.sf(statistic, 1))


def max_consecutive(breaches: pd.Series) -> int:
    """Longest run of consecutive breach days, as a plain-language read on clustering."""
    values = breaches.to_numpy()
    best = run = 0
    for v in values:
        run = run + 1 if v == 1 else 0
        best = max(best, run)
    return int(best)


def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
    alpha: float = DEFAULT_ALPHA,
) -> BacktestResult:
    """Run all three tests on a VaR series against what actually happened."""
    breaches = _breach_series(returns, var_series)
    n = len(breaches)
    if n == 0:
        raise ValueError("no overlapping days between the returns and the VaR series")

    n_breaches = int(breaches.sum())
    expected_rate = 1.0 - confidence

    kupiec_stat, kupiec_p = kupiec_test(n, n_breaches, expected_rate)
    ind_stat, ind_p = christoffersen_independence_test(breaches)

    # Conditional coverage is the sum of the two, jointly chi-squared with 2 d.f.
    joint_stat = kupiec_stat + ind_stat
    joint_p = float(stats.chi2.sf(joint_stat, 2))

    warnings: list[str] = []
    if n < LOW_POWER_OBSERVATIONS:
        warnings.append(
            f"only {n} observations. These tests have low power at this sample size, so "
            "'not rejected' is a weak statement: a mediocre model routinely survives them."
        )
    if n_breaches == 0:
        warnings.append(
            "no breaches at all. That is not a good result; it means the model is far too "
            "conservative, and capital held against a loss that never comes is capital doing "
            "nothing."
        )

    return BacktestResult(
        n_observations=n, n_breaches=n_breaches,
        expected_breaches=n * expected_rate,
        observed_rate=n_breaches / n, expected_rate=expected_rate,
        kupiec_statistic=kupiec_stat, kupiec_p=kupiec_p,
        christoffersen_statistic=ind_stat, christoffersen_p=ind_p,
        conditional_coverage_statistic=joint_stat, conditional_coverage_p=joint_p,
        max_consecutive_breaches=max_consecutive(breaches),
        alpha=alpha, warnings=tuple(warnings),
    )


def basel_traffic_light(n_breaches: int, n_observations: int = BASEL_WINDOW) -> dict:
    """The Basel supervisory zones: the crude test that has consequences attached.

    Defined for 250 trading days at 99%. Applied to a different window the zones are scaled
    proportionally, which is an approximation of the framework rather than the framework, and
    this says which one it is giving you.
    """
    scale = n_observations / BASEL_WINDOW
    green_max = BASEL_GREEN_MAX * scale
    yellow_max = BASEL_YELLOW_MAX * scale

    if n_breaches <= green_max:
        zone, meaning = "green", "no supervisory concern; the model is accepted as calibrated"
    elif n_breaches <= yellow_max:
        zone, meaning = "yellow", ("a capital multiplier is applied and the model is "
                                   "reviewed; the breach count is possible but improbable")
    else:
        zone, meaning = "red", ("the model is presumed flawed and must be revised; a breach "
                                "count this high is implausible under a correct model")

    return {
        "zone": zone, "meaning": meaning,
        "n_breaches": n_breaches, "n_observations": n_observations,
        "green_max": green_max, "yellow_max": yellow_max,
        "exact_window": n_observations == BASEL_WINDOW,
    }


def compare_backtests(
    returns: pd.Series,
    var_series_by_name: dict[str, pd.Series],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Backtest several VaR models against the same realised returns.

    The comparison is the deliverable. A single model's p-values say whether it survived; put
    an unconditional and a conditional model side by side and the table shows *which property*
    each one gets right, which is the thing worth knowing when choosing between them.
    """
    rows = []
    for name, series in var_series_by_name.items():
        result = backtest_var(returns, series, confidence)
        rows.append({
            "model": name,
            "breaches": result.n_breaches,
            "expected": round(result.expected_breaches, 1),
            "rate": result.observed_rate,
            "kupiec_p": result.kupiec_p,
            "christoffersen_p": result.christoffersen_p,
            "joint_p": result.conditional_coverage_p,
            "coverage": "pass" if result.passes_coverage else "FAIL",
            "independence": "pass" if result.passes_independence else "FAIL",
            "max_run": result.max_consecutive_breaches,
        })
    return pd.DataFrame(rows)
