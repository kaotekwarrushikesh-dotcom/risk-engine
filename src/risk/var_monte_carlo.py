"""Phase 6: Monte Carlo VaR.

Simulate a large number of possible return paths, then read the quantile off the simulated
distribution. Generated fresh on every run rather than stored, because a Monte Carlo result
that is cached and re-displayed is indistinguishable from a fixed number while looking like
a simulation.

**The draw method is the entire modelling decision, and it is exposed rather than buried.**

  normal        Draws from a fitted normal. Simulating from a distribution the data has
                already rejected (see Phase 5's Jarque-Bera test) produces a smooth,
                confident-looking distribution that inherits exactly the thin tail the
                normal assumption imposes. Included for comparison, not as a default.
  t             Draws from a Student-t with degrees of freedom estimated from the data,
                rescaled to unit variance. Keeps fat tails while remaining a parametric
                model, so it can produce a loss worse than any in the sample.
  bootstrap     Resamples the actual observed returns with replacement. Makes no shape
                assumption at all: the real skew, the real kurtosis and the real clustering
                of magnitudes are carried through because they are the data. This is the
                default here.

**What the bootstrap can and cannot do.** It samples from what happened, so like historical
VaR it cannot generate a single day worse than the worst observed one. What it *can* do that
Phase 4 cannot is build multi-day paths, where a ten-day loss is a sum of ten independently
drawn days and can therefore be far worse than any single historical day. So the bootstrap
lifts the one-day ceiling only at multi-day horizons, and this module says so rather than
implying simulation has escaped the sample.

**Independent draws discard volatility clustering, in the direction that flatters.** Real
markets have bad days that arrive in clusters; drawing each day independently breaks those
runs apart, so a simulated ten-day path is calmer than a real stressed fortnight. Every
multi-day result carries that warning. Phase 8's GARCH model is what addresses it.

**Simulation error is reported.** With 10,000 paths the 99% quantile is estimated from about
100 simulated observations, so the answer has its own noise on top of the estimation
uncertainty already in the inputs. The standard error across independent batches is computed
and reported, which is what makes "run more paths" a decision rather than a guess.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.risk.var_parametric import estimate_degrees_of_freedom, fit_distribution

DEFAULT_PATHS = 10_000
DEFAULT_SEED = 20250820
# Independent batches used to measure the simulation's own noise. Enough to give a stable
# standard error without multiplying the total work by more than a small factor.
STANDARD_ERROR_BATCHES = 10
# Below this many simulated observations beyond the quantile, the simulation is too small
# for the confidence level being asked of it.
MIN_SIMULATED_TAIL = 100

DrawMethod = str  # "bootstrap" | "normal" | "t"


@dataclass(frozen=True)
class MonteCarloVaRResult:
    """A simulated VaR, with the simulation's own uncertainty alongside it."""

    method: str
    draw_method: DrawMethod
    confidence: float
    horizon_days: int
    paths: int
    seed: int
    var_return: float
    var_value: float | None
    standard_error: float
    simulated_tail_observations: int
    worst_simulated_loss: float
    worst_observed_loss: float
    degrees_of_freedom: float | None = None
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def var_pct(self) -> float:
        return self.var_return * 100.0

    @property
    def exceeds_history(self) -> bool:
        """Whether the simulation produced a loss worse than anything actually observed."""
        return self.worst_simulated_loss > self.worst_observed_loss

    def summary(self) -> str:
        horizon = "1 day" if self.horizon_days == 1 else f"{self.horizon_days} days"
        line = (f"{self.confidence:.0%} {horizon} Monte Carlo VaR ({self.draw_method}): "
                f"{self.var_pct:.2f}% +/- {self.standard_error * 100:.2f}%")
        if self.var_value is not None:
            line += f" ({self.var_value:,.0f})"
        return line


def _to_simple_loss(log_return: float) -> float:
    return float(-(np.exp(log_return) - 1.0))


def simulate_paths(
    returns: pd.Series,
    paths: int,
    horizon_days: int,
    draw_method: DrawMethod,
    rng: np.random.Generator,
    degrees_of_freedom: float | None = None,
) -> np.ndarray:
    """Simulate `paths` cumulative log returns over `horizon_days`.

    Log returns are summed across the horizon rather than compounded, which is exactly why
    the engine works in log space: the sum of daily log returns *is* the multi-day log
    return, with no approximation.
    """
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        raise ValueError("at least two return observations are needed to simulate")

    if draw_method == "bootstrap":
        draws = rng.choice(clean, size=(paths, horizon_days), replace=True)
    elif draw_method == "normal":
        mu, sigma = float(np.mean(clean)), float(np.std(clean, ddof=1))
        draws = rng.normal(mu, sigma, size=(paths, horizon_days))
    elif draw_method == "t":
        if degrees_of_freedom is None:
            raise ValueError("degrees_of_freedom is required for the 't' draw method")
        mu, sigma = float(np.mean(clean)), float(np.std(clean, ddof=1))
        # Rescaled to unit variance for the same reason as in Phase 5: a standard t has
        # variance v/(v-2), so without this the simulated spread is counted twice.
        unit = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        draws = mu + rng.standard_t(degrees_of_freedom, size=(paths, horizon_days)) * unit * sigma
    else:
        raise ValueError(
            f"unknown draw method: {draw_method!r}, expected 'bootstrap', 'normal' or 't'"
        )

    return draws.sum(axis=1)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    paths: int = DEFAULT_PATHS,
    draw_method: DrawMethod = "bootstrap",
    seed: int = DEFAULT_SEED,
) -> MonteCarloVaRResult:
    """VaR from a simulated return distribution, regenerated on every call.

    Seeded by default so a figure can be discussed and reproduced; pass a different seed to
    confirm the answer is stable rather than an artefact of one draw, which is what the
    reported standard error quantifies directly.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")
    if paths < 100:
        raise ValueError(f"at least 100 paths are needed for a meaningful quantile, got {paths}")

    clean = returns.dropna()
    warnings: list[str] = []
    notes: list[str] = []
    tail = 1.0 - confidence

    degrees_of_freedom = estimate_degrees_of_freedom(clean) if draw_method == "t" else None
    rng = np.random.default_rng(seed)

    simulated = simulate_paths(clean, paths, horizon_days, draw_method, rng, degrees_of_freedom)
    quantile = float(np.quantile(simulated, tail))
    var_return = _to_simple_loss(quantile)

    # The simulation's own noise, measured by re-running in independent batches rather than
    # assumed to be negligible.
    batch_rng = np.random.default_rng(seed + 1)
    batch_size = max(paths // STANDARD_ERROR_BATCHES, 100)
    batch_estimates = [
        _to_simple_loss(float(np.quantile(
            simulate_paths(clean, batch_size, horizon_days, draw_method,
                           batch_rng, degrees_of_freedom), tail)))
        for _ in range(STANDARD_ERROR_BATCHES)
    ]
    standard_error = float(np.std(batch_estimates, ddof=1) / np.sqrt(STANDARD_ERROR_BATCHES))

    simulated_losses = -(np.exp(simulated) - 1.0)
    worst_simulated_loss = float(simulated_losses.max())
    simulated_tail = int((simulated <= quantile).sum())

    observed_worst_daily = float(clean.min())
    worst_observed_loss = _to_simple_loss(observed_worst_daily * horizon_days
                                          if horizon_days > 1 else observed_worst_daily)

    if simulated_tail < MIN_SIMULATED_TAIL:
        warnings.append(
            f"only {simulated_tail} simulated paths fall beyond the {confidence:.0%} quantile. "
            f"Run more paths: the answer is being read off {simulated_tail} points."
        )

    if draw_method == "normal":
        fit = fit_distribution(clean)
        if not fit.is_normal:
            warnings.append(
                "this simulation draws from a normal distribution the data itself rejects "
                f"(Jarque-Bera p = {fit.jarque_bera_p:.2g}). It will look smooth and confident "
                "and will inherit exactly the thin tail the assumption imposes."
            )

    if draw_method == "bootstrap":
        if horizon_days == 1:
            notes.append(
                "Resampling observed returns at a one-day horizon cannot produce a loss worse "
                "than the worst day in the sample, so this shares historical VaR's ceiling. "
                "The multi-day horizons are where resampling adds something."
            )
        else:
            notes.append(
                f"Each of the {horizon_days} days is drawn independently from the observed "
                "returns, so a simulated path can be far worse than any single historical day "
                "even though no individual day exceeds the sample."
            )

    if horizon_days > 1:
        warnings.append(
            "days are drawn independently, which breaks up the clustering real markets show: "
            "bad days arrive together, so a real stressed multi-day period is worse than this "
            "simulation implies. Phase 8's GARCH model is what addresses that."
        )

    notes.append(
        f"{paths:,} paths, seed {seed}, drawn by {draw_method}. Standard error of the estimate "
        f"across {STANDARD_ERROR_BATCHES} independent batches: {standard_error:.3%}."
    )

    return MonteCarloVaRResult(
        method="monte-carlo",
        draw_method=draw_method,
        confidence=confidence,
        horizon_days=horizon_days,
        paths=paths,
        seed=seed,
        var_return=var_return,
        var_value=var_return * portfolio_value if portfolio_value is not None else None,
        standard_error=standard_error,
        simulated_tail_observations=simulated_tail,
        worst_simulated_loss=worst_simulated_loss,
        worst_observed_loss=worst_observed_loss,
        degrees_of_freedom=degrees_of_freedom,
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def convergence_path(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    draw_method: DrawMethod = "bootstrap",
    path_counts: tuple[int, ...] = (500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000),
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """VaR against the number of simulated paths, to show where the answer settles.

    The point of running this once is to stop guessing at the path count. If the estimate is
    still moving at 10,000 paths, the number being quoted is partly simulation noise; if it
    flattened at 2,500, the extra paths are costing time and buying nothing.
    """
    rows = []
    dof = estimate_degrees_of_freedom(returns.dropna()) if draw_method == "t" else None
    for n in path_counts:
        rng = np.random.default_rng(seed)
        simulated = simulate_paths(returns.dropna(), n, horizon_days, draw_method, rng, dof)
        q = float(np.quantile(simulated, 1.0 - confidence))
        rows.append({"paths": n, "var_return": _to_simple_loss(q)})
    frame = pd.DataFrame(rows)
    frame["change_from_previous"] = frame["var_return"].diff().abs()
    return frame
