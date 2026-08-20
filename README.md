# Module 3: Live Institutional Risk Engine

> Quantifying market, fundamental and scenario risk through a live, interactive risk-management system.

**Status: Phases 1 to 8 and 12 of 14 built and tested.** Data ingestion with validation,
returns, volatility, beta, drawdown, historical/parametric/Monte Carlo VaR, Expected Shortfall,
GARCH(1,1) conditional volatility, and formal VaR backtesting all run and are tested against
both synthetic edge cases and real market history. Backtesting was pulled forward from its
roadmap position because it is what decides whether the GARCH work was worth it, and the
answer turned out to be a qualified yes rather than a clean one (see
[Phase 12](#phase-12-backtesting-and-whether-garch-was-actually-worth-it)). Fundamental and
valuation risk (reusing Modules 1 and 2), stress testing, portfolio risk, and the live
dashboard are not built yet. This README says so rather than implying a finished risk engine.
See [Roadmap](#roadmap).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest tests/ -q
```

## What "live" means here, precisely

yfinance is a free, unofficial interface to Yahoo Finance with no real-time service level
agreement. This engine never claims real-time data because of that: every fetch is
timestamped and classified as **DELAYED** or **STALE** by how old the data actually is
(`DataStatus.classification` in `data_loader.py`), never **LIVE**, regardless of how fresh a
given pull happens to be. "Live" in this module means *refreshable on demand and
recalculated from the current data*, not *real-time*, and the dashboard states that
distinction rather than implying more than the data source can support.

## Phase 1: data ingestion and validation

Fetches daily OHLCV, caches it to disk, and runs it through explicit validation before
anything downstream is allowed to touch it: duplicate dates, out-of-order dates, future
dates, non-positive or missing prices, non-trading (weekend) rows, gaps longer than a week,
extreme single-day moves, and a check that the data is not too sparse to support a daily
model. Issues are separated into three tiers: **issues** (corrected automatically, e.g. a
duplicate row), **warnings** (the calculation proceeds, but the result should be read with
the warning in mind), and **blocking** (the series is refused outright: *"Insufficient data
for reliable risk calculation"*). A blocking series never reaches a risk number; the two are
never allowed to disagree, because `clean_prices` applies exactly the corrections
`validate_prices` describes.

**Verified against two real, well-documented market events**, not only synthetic cases:

- NVIDIA's June 2024 10:1 stock split produces no false extreme-move flag, and checking the
  raw prices directly confirmed why: yfinance's `close` column is already split-adjusted
  regardless of the `auto_adjust` flag (only *dividends* create a close/adj_close gap), so
  the module's own comments describe what the data source actually does rather than what
  seemed plausible before checking.
- GameStop's May 2024 short-squeeze days are correctly flagged as extreme single-day moves,
  confirming the detector fires on real events and not only on injected test cases.

## Phase 2: returns and volatility

Both simple (`R_t = P_t/P_(t-1) - 1`) and log (`r_t = ln(P_t/P_(t-1))`) returns are
implemented, with log the default for anything that aggregates across time (volatility, VaR
horizon scaling, the GARCH model to come), since log returns are time-additive and simple
returns are not; simple returns remain the right choice for aggregating *across assets* in a
portfolio, which is a different kind of addition.

Volatility is annualised by the standard square-root-of-time scaling
(`sigma_annual = sigma_period x sqrt(periods_per_year)`), at daily, weekly, monthly and
annualised horizons, over a caller-supplied rolling window rather than a fixed constant,
since a 20-day window and a 252-day window answer different questions and neither is more
"correct." Rolling volatility and spike detection (episodes where volatility exceeds a
multiple of its own series median, grouped into contiguous episodes rather than listed day
by day) were checked against Apple's own recent volatility (annualised ~33%, consistent with
its known range) and correctly located an elevated-volatility episode in April-July 2025.

## Phase 3: beta and drawdown

Beta is computed two independent ways from the same aligned data
(`Cov(asset,market)/Var(market)` and OLS regression), and the two are asserted equal at
runtime: they are the same estimator by construction, so any gap between them is an
alignment bug, not a modelling difference, and the assertion catches it immediately rather
than after a bad beta has already fed a downstream figure. R-squared, correlation, a
t-statistic and a significance flag travel with the beta, because a beta with R-squared near
zero is a number that looks real and mostly is not; the module's own `confidence` field
downgrades a low-R-squared beta to "low" rather than presenting it at face value.

Checked against real data: NVIDIA and Tesla show betas of roughly 2.1-2.3 against the S&P
500 over the last three years (high, but so is their known volatility), while Coca-Cola
shows a beta near zero with R-squared near zero. That last result is initially
counterintuitive for a large defensive staple, so it was checked rather than assumed to be a
bug: the date alignment is correct (750 matched observations, identical index), the return
statistics are individually sane, and the moments/regression cross-check passes, so it is
reported as a genuinely low-confidence reading for this window rather than dismissed. That is
what the confidence field is for.

Drawdown (`(Value_t - Running Peak_t) / Running Peak_t`) tracks current and maximum
drawdown, duration since the last peak, and identifies peak-to-trough-to-recovery episodes
above a configurable depth threshold. Verified against the S&P 500's real, well-documented
history: the 2022 bear market shows as a -25.4% drawdown from January to October 2022,
recovering by January 2024, matching the real event; a second episode in February-April 2025
(-18.9%) lines up with the same period Phase 2's volatility-spike detector independently
flagged, which is a useful cross-check since the two were built and tested separately.

## Phase 4: historical VaR

    VaR(c) = the loss exceeded only (1 - c) of the time

Read straight off the empirical distribution: sort the observed returns, take the `1 - c`
quantile, report it as a positive loss. Nothing is assumed about the shape of the
distribution, so fat tails and skew are already in the answer rather than modelled into it.

**The defining limitation is asserted in a test, not just described.** Historical VaR is
bounded below by the worst loss in the sample: it has no mechanism for producing a number
worse than something that has already happened, so on a quiet sample it reports a small VaR
*because* nothing bad has occurred yet. Every result therefore carries the worst observed
loss alongside it, and raises a warning when the estimate sits within 90% of that worst
loss, which is the point at which the method is reporting the edge of its own data rather
than measuring a tail.

**How few observations a tail estimate rests on is reported, because the point estimate
hides it.** A 99% VaR on 250 trading days is a quantile supported by about 2.5 observations,
and on screen it looks exactly as precise as any other number. Each result carries
`tail_observations` (how many points actually sit beyond the quantile) and a bootstrap
confidence interval for the estimate itself, so "3.28%, interval 2.93% to 3.57%" can be told
apart from the same figure with a much wider one.

Two details that are quietly wrong in many implementations are handled explicitly. **Log
returns are converted before reporting** (`exp(r) - 1`), since a -5% log return is a 4.88%
loss, not a 5% one; this is immaterial at one day and material at ten. And **VaR is always a
positive number meaning a loss**, never a negative return, because a report that mixes the
two conventions is how a sign error reaches a decision.

Rolling VaR uses only the observations available up to each date, which matters because a
series contaminated by future data would pass any breach test trivially, and passing that
test is what Phase 12 exists to check.

**Verified against real market history**, on 10 years of daily data:

| | 95% 1-day VaR | 99% 1-day VaR | Worst day in sample | Breach rate (95%) |
|---|---|---|---|---|
| S&P 500 | 1.66% | 3.28% | -11.98% | 5.31% |
| Apple | 2.78% | 4.84% | -12.86% | 6.15% |
| GameStop | 6.63% | 13.86% | -60.00% | 5.31% |

The worst days are the real ones: the S&P 500's three deepest losses in ten years are all
March 2020 (16 March at -11.98%, its worst day since 1987), and GameStop's are 2 and 4
February and 28 January 2021, the collapse of the short squeeze. Breach rates against a
252-day rolling VaR land near the 5% the confidence level implies, which is the calibration
check: materially fewer breaches would mean a model too conservative rather than a safe one,
and both directions are failures.

## Phase 5: parametric VaR, and what the normal assumption actually costs

    Normal:     VaR(c) = -(mu + z_(1-c) * sigma)
    Student-t:  VaR(c) = -(mu + t_(1-c),v * sigma * sqrt((v - 2) / v))

Where Phase 4 reads a quantile off the data, this reads it off a fitted distribution. The
trade runs both ways: a model can produce a loss worse than anything in the sample, which
historical VaR structurally cannot, but only if the assumed shape is right.

**The assumption is tested, not asserted.** A Jarque-Bera test and the excess kurtosis run
before any normal VaR is reported, and a rejection produces a warning attached to the number
itself. Across ten years of daily data the assumption is rejected everywhere, decisively:

| | Excess kurtosis | Skew | Jarque-Bera p | Normal? |
|---|---|---|---|---|
| S&P 500 | 16.68 | -0.68 | ~0 | rejected |
| Apple | 6.60 | -0.11 | ~0 | rejected |
| GameStop | 44.94 | +0.82 | ~0 | rejected |

(A normal distribution has excess kurtosis of exactly 0.)

**The interesting result is that the error changes sign with the confidence level**, which
is not what "the normal model understates tail risk" alone would lead you to expect:

| | 95% historical | 95% normal | 99% historical | 99% normal |
|---|---|---|---|---|
| S&P 500 | 1.66% | **1.81%** | 3.28% | **2.57%** |
| Apple | 2.78% | **2.87%** | 4.84% | **4.08%** |
| GameStop | 6.63% | **10.53%** | 13.86% | **14.58%** |

At 95% the normal model is the *more* conservative of the two; at 99% it understates the
S&P's tail by 28% and Apple's by 19%. That is the signature of a fat-tailed distribution
rather than a contradiction: fitting a normal to a fat-tailed sample inflates sigma to
accommodate the extreme days, which pushes the moderate quantiles out too far while still
falling short in the far tail. GameStop is the extreme case, where the outliers are violent
enough that normal VaR overstates the 95% loss by more than half.

The practical consequence is that a normal VaR cannot be described as simply conservative or
simply optimistic; which one it is depends on where it is read. `normal_vs_historical_gap()`
reports the gap and its direction rather than assuming either.

**The Student-t variant estimates its degrees of freedom from the data** rather than taking
a conventional value, and the fits come back low (v of 2.5 to 3.1 across these three), which
is another read on how fat the tails are. One detail matters and is easy to omit: a standard
t with v degrees of freedom has variance `v / (v - 2)`, not 1, so the quantile is rescaled by
`sqrt((v - 2) / v)` before meeting sigma. Skipping that rescale double-counts the spread and
inflates every t-VaR; a test asserts the rescaled figure and checks the unscaled one would
have been larger.

Worth stating plainly: at 99% the t still lands below the historical figure for the S&P
(2.74% against 3.28%), so fitting a fatter distribution narrows the gap without closing it.
Mean and volatility are also scaled differently across horizons (linearly and by the square
root respectively), since scaling both by `sqrt(h)` is a common and quietly wrong shortcut.

## Phase 6: Monte Carlo VaR

Simulate many possible return paths, read the quantile off the simulated distribution.
Regenerated on every call rather than stored, since a cached Monte Carlo result is
indistinguishable from a fixed number while still looking like a simulation.

**The draw method is the whole modelling decision, so it is a parameter rather than a
default buried in the function.** Three are offered, and they behave exactly as the theory
says they should on 10 years of S&P 500 data at 99%, one day:

| Method | VaR | Simulation error | Worst simulated day | Beats history? |
|---|---|---|---|---|
| Historical (Phase 4) | 3.28% | — | — | — |
| Monte Carlo, bootstrap | 3.35% | ±0.066% | 11.98% | no |
| Monte Carlo, normal | 2.58% | ±0.017% | 5.2% | no |
| Monte Carlo, Student-t | 2.67% | ±0.066% | 18.4% | yes |

The bootstrap reproduces the historical figure to within about one standard error, which is
the check that resampling has not quietly altered the distribution it is drawing from. Its
worst simulated day is 11.98%, exactly the worst day in the sample, because resampling
cannot invent a day that never happened. The normal draw lands on the parametric normal
figure and produces a worst case of 5.2%, less than half the real worst day, which is what
the thin-tail assumption looks like when it is made visible. Only the Student-t draw exceeds
the historical record, at 18.4%, which is the one genuine advantage a parametric simulation
has over resampling.

**Where the bootstrap's ceiling lifts.** At one day, resampling shares historical VaR's
structural ceiling and the module says so rather than implying simulation has escaped the
sample. At ten days it does not: each day is drawn independently, so a path can compound
several bad draws into something far worse than any single historical day. On the S&P the
ten-day bootstrap VaR is 8.99% with a worst simulated path of 20.1%, against a worst single
observed day of 11.98%.

**Independent draws discard volatility clustering, in the direction that flatters.** Real
bad days arrive in runs; drawing each independently breaks those runs apart, so a simulated
fortnight is calmer than a real stressed one. Every multi-day result carries that warning,
and Phase 8's GARCH model is what addresses it.

**Simulation error is measured, not assumed negligible**, by re-running in independent
batches and reporting the standard error across them. `convergence_path()` tabulates VaR
against path count so the number of paths is a decision rather than a guess. On the S&P at
99% the estimate is still drifting slightly at 50,000 paths (0.0329 to 0.0335), which is
itself the finding: a far-tail quantile on fat-tailed data converges slowly, and quoting a
99% Monte Carlo VaR to three decimal places off 10,000 paths would be quoting noise.

## Phase 7: Expected Shortfall, and every method side by side

    ES(c) = E[loss | loss > VaR(c)]

VaR is a threshold and says nothing about what lies beyond it. Two portfolios can share an
identical 99% VaR while one loses a further 1% in the tail and the other loses everything,
and no VaR figure at any confidence level tells them apart. ES averages the losses past the
threshold instead of reporting where the threshold sits.

**ES is coherent, VaR is not, and the counterexample is built rather than cited.**
`demonstrate_var_subadditivity_failure()` constructs two independent positions, each losing
heavily with probability 4%. At 95% neither position's own tail reaches its loss, so each has
a VaR of -0.02 (a gain). The combined portfolio loses on either event, roughly 8% of the
time, which does reach into the 95% tail, so its VaR is 0.49 — larger than the sum of the
parts. VaR says combining two independent positions created risk. ES on the same data gives
0.51 against a sum of 1.56 and stays subadditive. This is why the Basel Committee moved
market-risk capital from 99% VaR to 97.5% ES.

**The full comparison, S&P 500, 99%, one day, 10 years of data:**

| Method | Assumption | VaR | ES | ES / VaR |
|---|---|---|---|---|
| Historical | none (empirical quantile) | 3.28% | **4.78%** | 1.45x |
| Parametric (normal) | returns are normal | 2.57% | **2.95%** | 1.15x |
| Parametric (Student-t) | Student-t, v fitted | 2.74% | **4.53%** | 1.65x |
| Monte Carlo (bootstrap) | resampled from observed | 3.35% | **4.83%** | 1.44x |
| Monte Carlo (Student-t) | simulated from fitted t | 2.67% | **4.24%** | 1.59x |

Two findings come out of that table that the VaR work alone did not show.

**The normal assumption fails worse on ES than on VaR.** Its VaR is 22% below the historical
figure; its ES is 38% below. That is not a coincidence of this sample: a normal distribution
has a fixed, thin tail shape, so its ES/VaR ratio is pinned near 1.15 at 99% regardless of
the data, while the real ratio here is 1.45. The error compounds precisely when you ask the
question ES exists to answer, because the model has no fat tail to average over. Apple shows
the same pattern (normal ES 4.67% against a historical 6.71%).

**The Student-t is a better model of the tail's shape than of its threshold.** It understates
VaR (2.74% against 3.28%) while getting ES nearly right (4.53% against 4.78%), and on Apple
it slightly overshoots (7.01% against 6.71%). A model can be wrong about where the tail
starts and right about how heavy it is, which is an argument for reporting both numbers
rather than choosing one.

The bootstrap rows reproduce the historical rows throughout, which is the consistency check
across the whole phase: five methods, three of them simulated, agreeing where they share
assumptions and diverging exactly where they do not.

**A bug this phase caught in its own code.** The tail was first selected as
`values <= quantile`, which looks equivalent to taking the worst `k` observations and is not.
On a distribution with an atom the comparison can select the entire sample, turning a "5% ES"
into the mean of everything, and nothing about the resulting number looks wrong. It surfaced
because the subadditivity demonstration reported ES as non-subadditive, which is
mathematically impossible and so could only be an implementation error. The tail is now
defined by rank, and a test pins the atom case directly. A second, smaller version of the same
class of bug lives in the rounding: `1 - 0.95` is `0.050000000000000044`, so
`ceil(1000 * (1 - 0.95))` is 51 rather than 50, and that one extra observation is the least
severe in the tail, dragging every ES toward the middle in the direction that understates risk.

## Phase 8: GARCH(1,1), and making the estimate conditional

    sigma^2_t = omega + alpha * e^2_(t-1) + beta * sigma^2_(t-1)

Everything in Phases 4 to 7 is **unconditional**. A 99% VaR of 3.28% for the S&P describes
the whole decade, blending 2017's calm with March 2020. It answers "how bad is a bad day for
this asset in general" and cannot answer "how bad is a bad day *this week*". GARCH makes
today's variance a blend of a long-run level, the size of yesterday's shock (`alpha`) and how
volatile things already were (`beta`), which is what reproduces the most robust empirical fact
about returns: volatility clusters. Every square-root-of-time scaling in the earlier phases
assumes precisely the opposite.

**The model is justified before it is fitted.** An ARCH-LM test checks whether there is
conditional heteroskedasticity to model at all, and a series without it gets a warning that
the GARCH is fitting noise rather than a silent fit. On the S&P the test returns p = 2.3e-167
before fitting and p = 0.314 on the standardised residuals afterwards, which is the pair worth
reporting together: there was overwhelming clustering, and the model absorbed it. Residual
diagnostics decide whether a fit is usable rather than convergence deciding for them.

The S&P fit: `alpha` 0.157, `beta` 0.841, persistence 0.998, half-life 345 trading days,
standardised shocks Student-t with 5.1 degrees of freedom. The variance model describes how
volatility moves and says nothing about the shape of the shocks, so that shape is fitted
separately and stays fat-tailed.

**A pathology this phase found and suppressed rather than displayed.** The model-implied
long-run volatility, `omega / (1 - persistence)`, came out at 48.6% annualised for the S&P
against a realised 18.1%. That figure is a faithful consequence of the fitted parameters and a
meaningless estimate, because the denominator is 0.002 and the ratio is numerically explosive
there: moving persistence from 0.995 to 0.999 swings the implied level from roughly 31% to
69%. Equity indices routinely fit at this persistence, so the case is normal rather than
exotic. The long-run figure is now flagged unreliable above 0.995 and suppressed from the
summary, and the volatility regime is judged against realised volatility instead, which is a
measurement rather than an extrapolation.

Forecasts mean-revert toward the long-run level at a rate set by persistence, so a forecast
made in a calm stretch rises and one made in a crisis falls. Multi-day volatility sums the
forecast daily variances rather than scaling one day by `sqrt(h)`, and since those terms are
unequal, that is exactly the behaviour square-root-of-time cannot represent.

## Phase 12: backtesting, and whether GARCH was actually worth it

Pulled forward from its roadmap position, because without it "GARCH is the better model" is
an appeal to authority rather than a measurement.

**A VaR model is judged on two independent properties**, and passing one proves nothing about
the other. *Unconditional coverage* (Kupiec) asks whether there are about the right **number**
of breaches. *Independence* (Christoffersen) asks whether they are **spread out** or arrive in
clusters. A model can produce exactly the right count and still be badly wrong if every breach
lands in the same fortnight, because that means it never adapted to the stress period and was
simply too high the rest of the time to compensate. That is the test an unconditional VaR is
expected to fail.

**Rolling one-day 99% VaR, backtested over 10 years, refitting GARCH every 50 days and using
only data available at each date:**

| | Breaches (exp. ~20) | Kupiec p | Christoffersen p | Coverage | Independence |
|---|---|---|---|---|---|
| **S&P 500** | | | | | |
| Historical (unconditional) | 25 | 0.2921 | **0.0030** | pass | **FAIL** |
| GARCH (conditional) | 29 | 0.0617 | 0.4373 | pass | pass |
| **Apple** | | | | | |
| Historical | 27 | 0.1429 | 0.3755 | pass | pass |
| GARCH | 21 | 0.8430 | 0.5055 | pass | pass |
| **Nifty 50** | | | | | |
| Historical | 21 | 0.7604 | **0.0189** | pass | **FAIL** |
| GARCH | 32 | **0.0101** | 0.5490 | **FAIL** | pass |

**The honest reading is not "GARCH wins".** What GARCH does reliably is fix the clustering:
the independence p-value improves on every series tested, and dramatically where it mattered
(S&P 0.0030 to 0.4373, Nifty 0.0189 to 0.5490). That is the property it was built to fix and
it fixes it consistently.

Its effect on the breach **count** is inconsistent. On Apple it improves coverage
substantially (0.1429 to 0.8430); on the S&P it degrades it while staying acceptable (0.2921
to 0.0617); on the Nifty it over-breaches badly enough to fail outright (32 against an
expected 20, p = 0.0101). So on the Nifty, GARCH trades a rejected independence test for a
rejected coverage test, and on the joint test it is the *worse* of the two models there
(0.0306 against 0.0608).

That result is reported rather than tuned away. The engine's own rule is that a model wrong in
a known direction, with the evidence for it, is more useful than one adjusted until it agrees,
and the Nifty case is a real limitation of this GARCH specification on that series, most
likely a symmetric model applied to a market whose downside shocks are not symmetric. A
GJR-GARCH or EGARCH variant, which lets negative shocks raise variance more than positive ones
of the same size, is the obvious next thing to test.

The Basel supervisory traffic light is included too, since it is the crude test with actual
consequences attached: on 250 days at 99%, up to 4 breaches is green, 5 to 9 yellow with a
capital multiplier, 10 or more red.

**One caveat stated in the output rather than left implicit**: at 250 observations these tests
have low power, so "not rejected" is a much weaker statement than it appears, and a mediocre
model routinely survives them. Sample size travels with every result for that reason.

## Structure

```text
risk_engine/
├── config/settings.py          benchmarks, windows, confidence levels, thresholds
├── data/                       cached price data (gitignored)
├── src/risk/
│   ├── data_loader.py          fetch, cache, validate, freshness classification
│   ├── returns.py              simple and log returns, annualisation
│   ├── volatility.py           historical/rolling volatility, regime, spike detection
│   ├── beta.py                 beta by two methods, cross-checked; rolling beta
│   ├── drawdown.py             drawdown series, summary, episode detection
│   ├── var_historical.py       empirical-quantile VaR, rolling VaR, breach counting
│   ├── var_parametric.py       normal and Student-t VaR, normality testing, method gap
│   ├── var_monte_carlo.py      simulated VaR, three draw methods, convergence and error
│   ├── expected_shortfall.py   ES per method, coherence demonstration, comparison table
│   ├── garch.py                GARCH(1,1), ARCH-LM, forecasting, conditional VaR
│   └── backtesting.py          Kupiec, Christoffersen, Basel traffic light, comparison
├── tests/
├── notebooks/
├── dashboard/
├── outputs/
└── requirements.txt
```

## Benchmarks

Verified live against yfinance before being listed, since a benchmark that silently returns
nothing would make every beta and relative-risk comparison built on it wrong without an
error: S&P 500 (`^GSPC`), STOXX Europe 600 (`^STOXX`), FTSE 100 (`^FTSE`), ISEQ (`^ISEQ`),
Nifty 50 (`^NSEI`), DAX (`^GDAXI`).

## Roadmap

- **Phase 9** Fundamental risk, reusing Module 1's ratios rather than recomputing them
- **Phase 10** Valuation risk and stress testing, reusing Module 2's DCF and WACC, including
  reverse stress testing against the current market price
- **Phase 11** Portfolio risk: correlation, risk contribution, concentration, Sharpe/Sortino
- **Phase 13** The live Streamlit dashboard tying all of the above together interactively
- **Phase 14** Integration with Modules 1 and 2 into one risk view

## Known limitations so far

1. **Square-root-of-time volatility scaling assumes i.i.d. returns**, which real returns are
   not; that is precisely why Phase 8 (GARCH) exists, to model the volatility clustering this
   simpler approach cannot.
2. **yfinance is an unofficial interface** with no service-level guarantee. Every fetch is
   classified DELAYED or STALE, never LIVE, and treated as capable of failing.
3. **Drawdown episode duration is in trading days**, which understates elapsed calendar time
   across holidays; this is a deliberate choice (trading days are what a rolling-window
   calculation actually consumes) rather than an oversight, but it means a duration figure
   should not be read as "calendar days out of the market."
4. **Beta confidence depends heavily on the window and the specific asset.** Coca-Cola's
   near-zero R-squared over the last three years is real and reported, not hidden, but it is
   also a reminder that a single point-in-time beta reading is not a permanent property of a
   company.
5. **The Phase 4 to 7 figures are unconditional; only Phase 8's are not.** A 99% VaR of
   3.28% for the S&P is an average-across-the-decade statement, and the number that matters
   in a stressed week is higher. GARCH conditional VaR is the one to use when the question is
   about now rather than in general, and the backtest in Phase 12 is the evidence for that
   rather than the assertion.
6. **Multi-day figures assume independent days**, in every method. Square-root-of-time
   scaling, parametric horizon scaling and Monte Carlo path simulation all break up the
   clustering real markets show, and all three err in the same direction: a real stressed
   fortnight is worse than any of them implies.
7. **Historical VaR and the bootstrap cannot exceed the worst observed day** at a one-day
   horizon. This is stated in the output rather than left for the reader to infer, but it
   means a quiet sample produces a reassuring number for a structural reason rather than an
   empirical one.
8. **Tail statistics rest on very few observations.** A 99% VaR on 250 days is supported by
   about 2.5 points and its ES by about 2. Counts and a bootstrap interval travel with each
   estimate for that reason, and neither makes the underlying sample any larger.
9. **The GARCH specification is symmetric, and that is a real limitation, not a footnote.**
   A GARCH(1,1) lets a shock raise variance without caring whether it was a gain or a loss,
   while equity downside shocks raise volatility more than upside ones of the same size. The
   Nifty backtest is where this shows: conditional VaR over-breaches there (32 against an
   expected 20, Kupiec p = 0.0101) and fails coverage outright. GJR-GARCH or EGARCH is the
   fix and is not built yet, so the Nifty conditional figures should be treated as the weakest
   in the engine.
10. **The model-implied long-run volatility is suppressed above 0.995 persistence**, because
   `omega / (1 - persistence)` is numerically explosive there and equity indices routinely fit
   into that range. Realised volatility is used as the regime benchmark instead. This is the
   right call and it does mean the engine reports no model-based view of where volatility
   settles in the long run for most index fits.
11. **Backtests at 250 observations have low power.** "Not rejected" is a much weaker claim
   than it looks, and a mediocre model routinely survives. Sample size travels with every
   backtest result for that reason, but no amount of reporting makes a short sample decisive.
