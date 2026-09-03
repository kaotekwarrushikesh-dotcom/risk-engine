# Module 3: Live Institutional Risk Engine

> Quantifying market, fundamental and scenario risk through a live, interactive risk-management system.

**Status: all 14 phases built and tested.** Data ingestion with validation,
returns, volatility, beta, drawdown, historical/parametric/Monte Carlo VaR, Expected Shortfall,
GARCH(1,1) conditional volatility, formal VaR backtesting, portfolio risk, fundamental risk
from Module 1's accounts, valuation risk and stress testing on Module 2's DCF, a composite view
combining all three, and an interactive dashboard over all of it run and are tested against
both synthetic edge cases and real market history. Backtesting was pulled forward from its
roadmap position because it is what decides whether the GARCH work was worth it, and the
answer turned out to be a qualified yes rather than a clean one (see
[Phase 12](#phase-12-backtesting-and-whether-garch-was-actually-worth-it)). See
[Phase 14](#phase-14-one-risk-view-built-from-all-three-engines) for what "complete" means
here and does not mean.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest tests/ -q
```

Run the interactive dashboard (any listed ticker, fetched live):

```bash
.venv/bin/streamlit run app.py
```

The engine is a pip-installable package, so other projects can depend on the calculations
without vendoring a copy of them:

```bash
pip install "git+https://github.com/kaotekwarrushikesh-dotcom/risk-engine.git"
```

```python
from risk_engine.garch import fit_garch
from risk_engine.backtesting import backtest_var
```

## What "live" means here, precisely

yfinance is a free, unofficial interface to Yahoo Finance with no real-time service level
agreement. This engine never claims real-time data because of that: every fetch is
timestamped and classified as **DELAYED** or **STALE** by how old the data actually is
(`DataStatus.classification` in `data_loader.py`), never **LIVE**, regardless of how fresh a
given pull happens to be. "Live" in this module means *refreshable on demand and
recalculated from the current data*, not *real-time*, and the dashboard states that
distinction rather than implying more than the data source can support. The dashboard shows
the fetch time and source as a quiet caption rather than a coloured badge, since the point is
that the claim is accurate, not that it is loud.

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

## Phase 11: portfolio risk

    sigma_p = sqrt(w' * Sigma * w)

Portfolio variance is a quadratic form, not a weighted average, and every result in this
phase follows from that one fact. Two holdings each at 25% volatility combine into something
less volatile than 25% unless they are perfectly correlated, and that reduction is the only
thing in finance that is genuinely free.

**Simple returns, not log returns.** The rest of the engine works in log returns because they
add across *time*. A portfolio return is the weighted sum of its holdings on the same day,
which is addition across *assets*, and log returns are not additive that way. The error is
small enough to pass inspection and grows with volatility, so the conversion is explicit and
a test pins the difference against the naive version.

**Risk contribution is the headline, because it is where intuition fails.** A holding's share
of portfolio risk is not its weight: it is weight times marginal contribution, which depends
on how the holding correlates with everything else. On an equally weighted portfolio of
Apple, Microsoft, Johnson & Johnson, Exxon and Coca-Cola, every weight is 20% and the risk
shares are not:

| Holding | Weight | Share of portfolio risk |
|---|---|---|
| Apple | 20% | **30.0%** |
| Microsoft | 20% | **26.0%** |
| Exxon | 20% | 19.9% |
| Coca-Cola | 20% | **13.0%** |
| Johnson & Johnson | 20% | **11.0%** |

Apple carries nearly three times Johnson & Johnson's risk for the same money. The
contributions sum exactly to portfolio volatility by Euler's theorem, which is asserted at
runtime rather than assumed: a mismatch means the decomposition is wrong, not imprecise.

A holding can also carry *negative* risk contribution when it moves against the rest of the
book, meaning it reduces total volatility rather than adding to it. No weight-based view of a
portfolio can show that.

**Diversification is measured, not asserted.** That same portfolio has a weighted average
volatility of 23.4% and an actual volatility of 14.2%, so diversification removed 39.2%. The
benefit shows in the tail too: portfolio VaR at 99% is 2.37% against 4.07% for the weighted
sum of the holdings' individual VaRs, because they do not all have their bad days together.

**Concentration gets a number rather than an eyeball.** The Herfindahl index and the effective
holdings count it implies say in one figure what a holdings count cannot: ten positions where
one is 85% of the book has an effective count below two, and the app says so.

**Minimum variance is offered and the rest of the frontier is not.** It is the one point on
the efficient frontier that needs no return forecast, and expected returns are estimated with
far more error than covariances and would dominate any other point. On the portfolio above it
moves weight from Apple (1.3%) into Johnson & Johnson (35.0%) and Coca-Cola (32.6%), taking
volatility from 14.2% to 12.6%.

**Sharpe and Sortino disagree on purpose.** Sharpe penalises upside volatility exactly as much
as downside; Sortino uses downside deviation only. Where they diverge the return distribution
is asymmetric, and the app says so rather than leaving the reader to compare two numbers.

## Phases 9 and 10: fundamental and valuation risk

These are the phases where the three engines meet. Neither recomputes anything: Phase 9 reads
Module 1's ratios and Phase 10 shocks Module 2's DCF, so there is one implementation of each
calculation and no second copy free to drift from it. Both are optional dependencies
(`pip install "risk-engine[integration]"`), because the market-risk core stands on its own and
a consumer who wants VaR should not be made to install two other engines to get it.

### Phase 9: fundamental risk

Every earlier phase measures **market** risk, meaning how much the price moves. This measures
**fundamental** risk, meaning how fragile the business is. The interesting cases are where
they disagree, and the engine names the divergence:

| | Fundamental risk | Market risk | Verdict |
|---|---|---|---|
| Apple | 41 | 32 | the two broadly agree |
| Pfizer | **52** | 26 | **fundamental risk exceeds market risk** |
| Tesla | 25 | **100** | **market risk exceeds fundamental risk** |

Pfizer is the case worth dwelling on, and it is the more dangerous of the two divergences: a
calm share price is not evidence about the balance sheet, and volatility cannot see leverage
building under a quiet stock. The flags say what the score cannot: interest cover falling 7.4x
per year and EBITDA margin compressing 4.4 points per year over the last four years, which is
the real post-pandemic decline showing up in the accounts.

**The question asked here is deliberately different from Module 1's.** Module 1 scores
financial *health*, which is mostly profitability. This scores financial *risk*, and the two
diverge in three specific ways:

- **Profitability barely features**, because a highly profitable company can still be fragile.
- **Direction outweighs level.** A company at 2.5x interest cover that was 8x three years ago
  is more dangerous than one that has sat at 2.5x throughout. Module 1 reads the latest year
  and cannot tell them apart; this measures the regression slope over recent years, so a
  single unusual year cannot decide whether a company is called deteriorating.
- **Earnings quality gets its own pillar**, since profit not backed by cash is the most useful
  early warning in published accounts and is invisible in a margin.

Wording is deliberately careful. A ratio cannot prove misconduct, so a persistent gap between
profit and cash is described as an earnings-quality question worth investigating rather than a
conclusion.

### Phase 10: valuation risk and reverse stress testing

Valuation risk is not "the DCF says this is expensive". That is a view, and a view is not a
risk. It is **how far the answer moves when an assumption moves**, and the two are close to
independent: a valuation can sit exactly on the market price and still be worthless if a
quarter-point change in the discount rate moves it by half.

On Apple: 70% of enterprise value sits in the terminal value, a 25bp change in WACC moves the
implied price by 3.5%, and a 50bp change in terminal growth moves it by 4.7%. The sensitivity
grid reports both assumptions at once, since a one-at-a-time table hides that they interact.

**Reverse stress testing is the part worth having.** Rather than asking what the company is
worth, it asks what today's price requires and leaves the reader to judge whether that is
plausible. This is much harder to fool yourself with, because the output is a required
assumption rather than a comfortable answer. For Apple the market price implies a discount
rate of 4.33% against the model's 9.35%, and at the model's own WACC it requires terminal
growth of **7.62% in perpetuity**, which is far above nominal GDP growth and therefore
something no business has ever sustained. That is a much sharper statement than "the DCF reads
low".

Module 2's documented calibration bias applies to every level here and is attached to the
result rather than left in a README. It matters less than it looks: a bias that moves every
scenario in the same direction affects the level and not the sensitivity, which is what this
phase measures, and a test pins that a uniform level shift leaves the sensitivity unchanged.

Stress scenarios shock the discount rate and terminal outlook, which this engine can move
rigorously. A revenue or margin shock belongs in Module 2's own scenario engine, which rebuilds
the forecast properly, and a cruder copy here would be worse than not having one.

## Phase 14: one risk view, built from all three engines

Phases 1 to 8 answer how much the price moves. Phase 9 answers how fragile the business is.
Phase 10 answers how much the valuation depends on its assumptions. Nothing before this phase
put the three side by side, and each is computed from a different source: this engine's own
market data, Module 1's filed accounts, and Module 2's DCF. This phase composes them.

**The composite score is the least interesting number this phase produces.** A single figure
that blends three independent readings is a worse answer than any one of the three read on its
own, because it can hide exactly the disagreement worth knowing about. The useful output is
naming *where* the three disagree, and each kind of disagreement is a different warning:

| Disagreement | What it means |
|---|---|
| Fundamental risk exceeds market risk | A quiet share price sitting over a weakening balance sheet. Price volatility cannot see leverage building, so calm is not evidence the accounts support (Phase 9's own finding, reused here rather than re-derived) |
| Valuation is fragile and the volatility regime is elevated | The inputs a sensitive DCF depends on (beta, the discount rate) are least stable exactly when the valuation is most sensitive to them |
| Fundamentals are deteriorating and the terminal value dominates | Most of the valuation rests on a perpetuity assumption about a business whose own recent trend is the assumption's biggest risk |

**Verified live, and the three cases came back exactly as the earlier phases would predict:**

| Ticker | Market | Fundamental | Valuation | Composite | Divergence found |
|---|---|---|---|---|---|
| Apple | 37 | 41 | 60 | 45 (Moderate) | none |
| Pfizer | 49 | 52 | 75 | 58 (Elevated) | **two, both critical/elevated** |
| Tesla | 96 | 25 | 52 | 62 (Elevated) | market exceeds fundamental (note) |

Pfizer fires both flags that matter: fundamental risk (52) exceeding market risk (26 on the
same volatility-only basis Phase 9 uses), and 71% of enterprise value sitting in a terminal
value that assumes trend continuation while fundamental deterioration scores 67/100. Both
reproduce, independently, the same finding Phase 9 reached on its own when it first flagged
Pfizer's post-pandemic decline. Tesla shows the opposite pattern: extreme market risk (96) far
above a modest fundamental score (25), the volatile-share-over-a-solid-business case rather
than the dangerous one. Apple shows genuine agreement across all three, and correctly raises no
divergence at all rather than manufacturing one.

**Every dimension is put on the same 0-100, higher-is-riskier scale**, matching Module 1's
health score and Phase 9's own convention. Market risk is built from volatility, 99% VaR, max
drawdown and beta, each floor/ceiling scored and combined by weight, the identical pattern
Phase 9 and Module 1 both use. A low-confidence beta (R-squared near zero) is excluded rather
than trusted at face value, the same standard Phase 3 itself applies to a beta reading. The
current GARCH volatility regime nudges the market score by a bounded amount rather than
entering as a fifth weighted metric, since it is context for reading the other four rather than
an independent measurement.

Valuation risk is mapped onto the same scale from three components: how much of the value sits
beyond the forecast (assumption dependence), how far a 25 basis point discount-rate move swings
the answer (fragility), and how far the model's own WACC sits from the market-implied rate
Module 2 solves for. The third is excluded, and says so, when no market-implied rate could be
solved, which happens when the DCF and the price are too far apart for the search range to
reach.

**A dimension that is unavailable is dropped and the remaining weights renormalised**, not
penalised. An index has no filed statements and no EBITDA to run a DCF on, so `^GSPC` returns a
composite that is the market score alone, with `fundamental` and `valuation` both listed as
excluded rather than scored as safe. This is Module 1's own rule for a missing metric, applied
here to a missing dimension: a gap in the data must never be read as a clean bill of health.

Every function that scores or combines is pure, taking already-computed inputs, so the
composition logic is tested without a network call; only `assess()`, which fetches live data
and calls Phases 9 and 10, needs one.

## Structure

```text
risk_engine/
├── app.py                      the interactive dashboard
├── data/                       cached price data (gitignored)
├── risk_engine/                the installable package
│   ├── settings.py             benchmarks, windows, confidence levels, thresholds
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
│   ├── backtesting.py          Kupiec, Christoffersen, Basel traffic light, comparison
│   ├── portfolio.py            correlation, risk contribution, concentration, Sharpe/Sortino
│   ├── fundamental.py          balance-sheet fragility from Module 1's ratios
│   ├── valuation_risk.py       sensitivity and reverse stress testing on Module 2's DCF
│   └── composite.py            market, fundamental and valuation risk on one scale
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

All 14 phases are built. Ideas beyond the original scope, not committed to:

- Stress testing against named historical scenarios (2008, 2020, 2022) rather than parametric
  shocks alone
- A saved-portfolio mode, so the composite view runs across a book rather than one ticker at a
  time
- Asymmetric GARCH (GJR-GARCH or EGARCH), which Phase 8's own findings named as the fix for the
  Nifty backtest's coverage failure

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
12. **The composite's default weights (market 40%, fundamental 30%, valuation 30%) are a
   stated choice, not a derived one.** Nothing in the data determines how much market risk
   should count against fundamental risk; a different, equally defensible analyst could
   reasonably weight them differently. `compose()` accepts custom weights precisely because
   the default should not be mistaken for a discovered fact.
13. **The divergence checks are the three that seemed most worth building, not an exhaustive
   set.** Other genuine disagreements between the three dimensions are possible and are not
   yet checked for; the three implemented are the ones with a clear, testable real-world
   reading rather than every combination the data happens to allow.
