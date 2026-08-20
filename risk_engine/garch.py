"""Phase 8: GARCH(1,1) conditional volatility.

    sigma^2_t = omega + alpha * e^2_(t-1) + beta * sigma^2_(t-1)

Everything built up to here is **unconditional**: a 99% VaR of 3.28% for the S&P describes
the distribution of returns across the whole decade, blending 2017's calm with March 2020.
It is the right answer to "how bad is a bad day for this asset in general" and the wrong
answer to "how bad is a bad day *this week*", because it carries no information about what
volatility is doing right now.

GARCH makes the estimate conditional. Today's variance is a weighted blend of three things:
a long-run average (`omega`), how large yesterday's shock was (`alpha`, the reaction term),
and how volatile things already were (`beta`, the persistence term). That structure is what
reproduces the single most robust empirical fact about returns: **volatility clusters.**
Large moves follow large moves, of either sign. Every square-root-of-time scaling in Phases
4 to 7 assumes exactly the opposite, and this module is what replaces that assumption.

**GARCH is justified before it is fitted, not after.** An ARCH-LM test on the squared
residuals checks whether there is conditional heteroskedasticity to model in the first place.
If a series shows no ARCH effects, a GARCH model fits noise and the extra machinery buys
nothing, so `fit_garch` reports the test rather than assuming the answer.

**alpha + beta is the number to read.** Their sum is persistence: how long a volatility shock
takes to decay. Near 1 means shocks fade slowly and today's turbulence still matters in a
month; the implied half-life makes that concrete. At exactly 1 the model is IGARCH, variance
is non-stationary, no long-run average exists, and the unconditional variance the model
implies is undefined rather than large. Equity series routinely fit at 0.98 to 0.99, close
enough to that boundary that the distinction is worth flagging rather than passing over.

**Residual diagnostics decide whether the fit is usable.** If the model has captured the
clustering, its standardised residuals should have no ARCH effects left in them. Any
remaining ones mean the model is misspecified, so the diagnostic is run and reported instead
of the fit being trusted because it converged.

**The distribution of the standardised residuals is a separate choice from the variance
model.** GARCH describes how variance moves; it says nothing about the shape of the shocks.
Fitting Gaussian errors to equity returns leaves fat tails in the residuals and understates
VaR for the same reason Phase 5's normal VaR does, so a Student-t is the default here and the
fitted degrees of freedom are reported.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252
# The `arch` package works better on percent-scale returns; fitting on raw decimals gives a
# tiny omega and can stall the optimiser.
SCALE = 100.0
# Above this, persistence is close enough to the non-stationary boundary to say so.
HIGH_PERSISTENCE = 0.98
# Above this, omega / (1 - persistence) is numerically explosive and the long-run level it
# implies should not be quoted. See `long_run_is_reliable`.
UNRELIABLE_LONG_RUN_PERSISTENCE = 0.995
# ARCH-LM below this p-value means conditional heteroskedasticity is present.
ARCH_ALPHA = 0.05
MIN_OBSERVATIONS = 250


@dataclass(frozen=True)
class ArchTestResult:
    """Whether a series has ARCH effects worth modelling."""

    statistic: float
    p_value: float
    lags: int

    @property
    def has_arch_effects(self) -> bool:
        return self.p_value < ARCH_ALPHA

    def reading(self) -> str:
        if self.has_arch_effects:
            return (
                f"ARCH effects present (LM p = {self.p_value:.2g}): variance depends on recent "
                "shocks, so there is clustering for a GARCH model to capture."
            )
        return (
            f"No ARCH effects detected (LM p = {self.p_value:.3f}). A GARCH model would be "
            "fitting noise here, and the constant-volatility assumption behind Phases 4 to 7 "
            "is not doing any harm on this series."
        )


@dataclass(frozen=True)
class GarchFit:
    """A fitted GARCH(1,1), with the diagnostics that decide whether to believe it."""

    omega: float
    alpha: float
    beta: float
    nu: float | None
    distribution: str
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    # Volatility actually observed over the fitted sample. Kept as the benchmark for judging
    # the current regime, because it is a measurement rather than an extrapolation and does
    # not blow up near the persistence boundary the way the model-implied long-run level does.
    realised_annualised_volatility: float
    conditional_volatility: pd.Series
    standardised_residuals: pd.Series
    arch_test_before: ArchTestResult
    arch_test_after: ArchTestResult
    converged: bool
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def persistence(self) -> float:
        """alpha + beta. How slowly a volatility shock decays."""
        return self.alpha + self.beta

    @property
    def is_stationary(self) -> bool:
        """Variance mean-reverts only if persistence is below 1."""
        return self.persistence < 1.0

    @property
    def half_life_days(self) -> float:
        """Trading days for a volatility shock to decay halfway back to the long-run level."""
        p = self.persistence
        if p <= 0 or p >= 1:
            return float("inf")
        return float(np.log(0.5) / np.log(p))

    @property
    def long_run_variance(self) -> float:
        """omega / (1 - alpha - beta), undefined without stationarity."""
        if not self.is_stationary:
            return float("nan")
        return self.omega / (1.0 - self.persistence)

    @property
    def long_run_is_reliable(self) -> bool:
        """Whether the implied long-run level is worth quoting at all.

        `omega / (1 - persistence)` divides by a number approaching zero, so near the
        non-stationary boundary the implied long-run volatility is numerically explosive: on
        a real S&P fit, moving persistence from 0.995 to 0.999 swings it from 31% to 69%
        annualised, against a realised sample volatility of about 18%. The figure is a
        faithful consequence of the parameters and still a meaningless estimate, so it is
        flagged rather than displayed as though the model had measured something.
        """
        return self.is_stationary and self.persistence <= UNRELIABLE_LONG_RUN_PERSISTENCE


    @property
    def long_run_annualised_volatility(self) -> float:
        lrv = self.long_run_variance
        if np.isnan(lrv):
            return float("nan")
        return float(np.sqrt(lrv) / SCALE * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def current_annualised_volatility(self) -> float:
        return float(self.conditional_volatility.iloc[-1] / SCALE * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def residuals_are_clean(self) -> bool:
        """Whether the fit removed the clustering it was meant to capture."""
        return not self.arch_test_after.has_arch_effects

    def summary(self) -> str:
        benchmark = (f"a model-implied long-run {self.long_run_annualised_volatility:.1%}"
                     if self.long_run_is_reliable
                     else f"a realised {self.realised_annualised_volatility:.1%} "
                          "(the model-implied long-run level is unreliable at this persistence)")
        return (
            f"GARCH(1,1)-{self.distribution}: omega={self.omega:.4g}, alpha={self.alpha:.3f}, "
            f"beta={self.beta:.3f}, persistence={self.persistence:.4f} "
            f"(half-life {self.half_life_days:.0f} days). Current annualised volatility "
            f"{self.current_annualised_volatility:.1%} against {benchmark}."
        )


@dataclass(frozen=True)
class VolatilityForecast:
    """A forward volatility path, and what it converges to."""

    horizon_days: int
    daily_volatility: pd.Series
    annualised_volatility: pd.Series
    cumulative_volatility: float
    long_run_annualised: float
    notes: tuple[str, ...] = ()


def arch_lm_test(residuals: np.ndarray, lags: int = 10) -> ArchTestResult:
    """Engle's ARCH-LM test: regress squared residuals on their own lags.

    If recent squared shocks predict today's squared shock, variance is not constant. The
    test statistic is `n * R^2` from that regression, distributed chi-squared with `lags`
    degrees of freedom under the null of no ARCH effects.
    """
    e2 = residuals ** 2
    n = len(e2)
    if n <= lags + 1:
        raise ValueError(f"need more than {lags + 1} observations for an ARCH-LM test at {lags} lags")

    # Build the lag matrix, with a constant.
    y = e2[lags:]
    X = np.column_stack([np.ones(len(y))] + [e2[lags - i: -i] for i in range(1, lags + 1)])

    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefficients
    ss_residual = float(np.sum((y - fitted) ** 2))
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0

    statistic = len(y) * r_squared
    p_value = float(stats.chi2.sf(statistic, lags))
    return ArchTestResult(statistic=float(statistic), p_value=p_value, lags=lags)


def fit_garch(
    returns: pd.Series,
    distribution: str = "t",
    lags: int = 10,
) -> GarchFit:
    """Fit GARCH(1,1), re-estimated on every call rather than cached.

    `distribution` is the assumed shape of the standardised shocks: "t" (default, since
    equity residuals stay fat-tailed after the variance model is applied) or "normal".
    """
    from arch import arch_model

    if distribution not in ("t", "normal"):
        raise ValueError(f"unknown distribution: {distribution!r}, expected 't' or 'normal'")

    clean = returns.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        raise ValueError(
            f"GARCH needs at least {MIN_OBSERVATIONS} observations to estimate three parameters "
            f"plus a tail, got {len(clean)}"
        )

    scaled = clean * SCALE
    warnings: list[str] = []
    notes: list[str] = []
    realised = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Justify the model before fitting it: demeaned returns are the residuals to test.
    before = arch_lm_test((scaled - scaled.mean()).to_numpy(), lags=lags)
    if not before.has_arch_effects:
        warnings.append(
            f"no ARCH effects were detected before fitting (LM p = {before.p_value:.3f}), so this "
            "GARCH model is fitting noise. The constant-volatility assumption behind the earlier "
            "phases is adequate for this series."
        )

    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist=distribution)
    fitted = model.fit(disp="off", show_warning=False)

    params = fitted.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    nu = float(params["nu"]) if distribution == "t" and "nu" in params else None

    conditional_volatility = pd.Series(fitted.conditional_volatility, index=clean.index)
    standardised = pd.Series(fitted.std_resid, index=clean.index).dropna()

    after = arch_lm_test(standardised.to_numpy(), lags=lags)

    persistence = alpha + beta
    if persistence >= 1.0:
        warnings.append(
            f"persistence (alpha + beta) is {persistence:.4f}, at or above 1. Variance is "
            "non-stationary: shocks never fully decay, there is no long-run average level to "
            "revert to, and the implied unconditional variance is undefined rather than large. "
            "Forecasts from this fit trend rather than converge."
        )
    elif persistence > HIGH_PERSISTENCE:
        notes.append(
            f"Persistence is {persistence:.4f}, close to the non-stationary boundary. Shocks "
            f"decay slowly: half-life {np.log(0.5) / np.log(persistence):.0f} trading days, so "
            "today's turbulence still matters months out. This is typical for equity indices "
            "rather than a sign of a bad fit."
        )

    if persistence > UNRELIABLE_LONG_RUN_PERSISTENCE and persistence < 1.0:
        implied = np.sqrt(omega / (1.0 - persistence)) / SCALE * np.sqrt(TRADING_DAYS_PER_YEAR)
        warnings.append(
            f"the model-implied long-run volatility of {implied:.1%} should not be quoted. "
            f"omega / (1 - persistence) divides by {1.0 - persistence:.4f}, so the figure is "
            "numerically explosive here: on this kind of fit, moving persistence from 0.995 to "
            "0.999 swings the implied level from roughly 31% to 69% annualised. The realised "
            f"{realised:.1%} is the measurement to compare against instead."
        )

    if after.has_arch_effects:
        warnings.append(
            f"ARCH effects remain in the standardised residuals (LM p = {after.p_value:.2g}). The "
            "model has not captured all the clustering, so it is misspecified: a higher order or "
            "an asymmetric variant (GJR-GARCH, EGARCH) would be the next thing to try."
        )
    else:
        notes.append(
            f"Standardised residuals show no remaining ARCH effects (LM p = {after.p_value:.3f}), "
            "so the variance model has absorbed the clustering it was fitted to capture."
        )

    if nu is not None:
        notes.append(
            f"Standardised shocks follow a Student-t with {nu:.1f} degrees of freedom. The "
            "variance model describes how volatility moves; it says nothing about the shape of "
            "the shocks, so that shape is fitted separately and stays fat-tailed here."
        )

    return GarchFit(
        omega=omega, alpha=alpha, beta=beta, nu=nu, distribution=distribution,
        log_likelihood=float(fitted.loglikelihood),
        aic=float(fitted.aic), bic=float(fitted.bic),
        n_observations=len(clean),
        realised_annualised_volatility=realised,
        conditional_volatility=conditional_volatility,
        standardised_residuals=standardised,
        arch_test_before=before, arch_test_after=after,
        converged=bool(fitted.convergence_flag == 0),
        warnings=tuple(warnings), notes=tuple(notes),
    )


def forecast_volatility(fit: GarchFit, horizon_days: int = 10) -> VolatilityForecast:
    """Forecast conditional volatility forward, which is what GARCH is actually for.

    The forecast mean-reverts toward the long-run level at a rate set by persistence, so a
    forecast made in a calm period rises and one made in a crisis falls. That is the whole
    behaviour square-root-of-time scaling cannot produce: it projects today's volatility
    unchanged, forever.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")

    last_variance = float(fit.conditional_volatility.iloc[-1]) ** 2
    long_run = fit.long_run_variance if fit.is_stationary else float("nan")

    variances = []
    variance = last_variance
    for _ in range(horizon_days):
        # One-step recursion under the expectation E[e^2] = sigma^2:
        #   E[sigma^2_(t+1)] = omega + (alpha + beta) * sigma^2_t
        variance = fit.omega + fit.persistence * variance
        variances.append(variance)

    daily = pd.Series(np.sqrt(variances) / SCALE, index=range(1, horizon_days + 1))
    annualised = daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Multi-day volatility is the square root of summed daily variances, which is where GARCH
    # and square-root-of-time genuinely part company: the terms being summed are not equal.
    cumulative = float(np.sqrt(np.sum(np.array(variances)) ) / SCALE)

    notes = [
        "Each step applies E[sigma^2_(t+1)] = omega + (alpha + beta) * sigma^2_t, so the path "
        "mean-reverts toward the long-run level rather than holding today's volatility fixed.",
        "Cumulative volatility sums the forecast daily variances rather than scaling one day by "
        "sqrt(h); the terms are unequal, which is exactly what sqrt-of-time cannot represent.",
    ]
    if not fit.is_stationary:
        notes.append(
            "This fit is non-stationary, so the forecast trends away rather than converging and "
            "the long-run level it would revert to does not exist."
        )

    return VolatilityForecast(
        horizon_days=horizon_days,
        daily_volatility=daily,
        annualised_volatility=annualised,
        cumulative_volatility=cumulative,
        long_run_annualised=fit.long_run_annualised_volatility,
        notes=tuple(notes),
    )


def conditional_var(
    fit: GarchFit,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
) -> dict:
    """VaR from today's conditional volatility rather than the whole sample's.

    This is the payoff of the phase. The quantile comes from the fitted shock distribution
    and the scale from the GARCH forecast, so the number moves with the volatility regime:
    higher than unconditional VaR in a crisis, lower in a calm stretch. An unconditional VaR
    is wrong in both directions at different times and right on average, which is the least
    useful place for a risk number to be right.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")

    forecast = forecast_volatility(fit, horizon_days)
    sigma = forecast.cumulative_volatility
    tail = 1.0 - confidence

    if fit.distribution == "t" and fit.nu is not None:
        # Same unit-variance rescaling as Phase 5, for the same reason.
        unit = np.sqrt((fit.nu - 2.0) / fit.nu)
        quantile = float(stats.t.ppf(tail, fit.nu)) * unit
        shock_note = f"Student-t quantile with {fit.nu:.1f} degrees of freedom, rescaled to unit variance."
    else:
        quantile = float(stats.norm.ppf(tail))
        shock_note = "Normal quantile."

    var_return = float(-(np.exp(quantile * sigma) - 1.0))

    # The regime is judged against realised volatility, not the model-implied long-run level,
    # which is numerically unstable at the persistence equity series typically fit at.
    benchmark = fit.realised_annualised_volatility
    current = fit.current_annualised_volatility

    return {
        "var_return": var_return,
        "var_value": var_return * portfolio_value if portfolio_value is not None else None,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "conditional_sigma": sigma,
        "current_annualised_volatility": current,
        "realised_annualised_volatility": benchmark,
        "long_run_annualised_volatility": (fit.long_run_annualised_volatility
                                           if fit.long_run_is_reliable else float("nan")),
        "regime": (
            "elevated" if current > benchmark * 1.25
            else "subdued" if current < benchmark * 0.8
            else "normal"
        ),
        "notes": (shock_note,) + forecast.notes,
    }


def rolling_conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
    refit_every: int = 50,
    min_window: int = 500,
    distribution: str = "t",
) -> pd.Series:
    """A one-day conditional VaR series, for backtesting against realised losses.

    The model is refitted periodically rather than every day, which is what a desk actually
    does: re-estimating three parameters on 500+ observations daily is expensive and moves
    them very little. Between refits the variance recursion is still updated with each new
    return, so the *volatility* is genuinely daily even though the *parameters* are not.

    Crucially each estimate uses only data up to that day. A conditional VaR series fitted on
    the whole sample would know about future crises and would pass any backtest trivially,
    which is precisely the error Phase 12 exists to detect.
    """
    clean = returns.dropna()
    if len(clean) <= min_window:
        raise ValueError(f"need more than {min_window} observations, got {len(clean)}")

    values = (clean * SCALE).to_numpy()
    out = pd.Series(np.nan, index=clean.index)

    omega = alpha = beta = nu = None
    variance = None

    for i in range(min_window, len(clean)):
        if (i - min_window) % refit_every == 0:
            fit = fit_garch(clean.iloc[:i], distribution=distribution)
            omega, alpha, beta, nu = fit.omega, fit.alpha, fit.beta, fit.nu
            variance = float(fit.conditional_volatility.iloc[-1]) ** 2
            mean = float(values[:i].mean())
        else:
            shock = values[i - 1] - mean
            variance = omega + alpha * shock**2 + beta * variance

        next_variance = omega + (alpha + beta) * variance
        sigma = np.sqrt(next_variance) / SCALE

        if distribution == "t" and nu is not None:
            quantile = float(stats.t.ppf(1 - confidence, nu)) * np.sqrt((nu - 2.0) / nu)
        else:
            quantile = float(stats.norm.ppf(1 - confidence))

        out.iloc[i] = -(np.exp(quantile * sigma) - 1.0)

    return out
