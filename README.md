# Module 3: Live Institutional Risk Engine

> Quantifying market, fundamental and scenario risk through a live, interactive risk-management system.

**Status: Phases 1 to 3 of 14 built and tested.** Data ingestion with validation, returns,
historical and rolling volatility, beta, and drawdown all run and are tested against both
synthetic edge cases and real market history. VaR, Expected Shortfall, GARCH, fundamental
and valuation risk (reusing Modules 1 and 2), stress testing, portfolio risk, backtesting,
and the live dashboard are not built yet. This README says so rather than implying a
finished risk engine. See [Roadmap](#roadmap).

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
│   └── drawdown.py             drawdown series, summary, episode detection
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

- **Phase 4** Historical VaR
- **Phase 5** Parametric VaR
- **Phase 6** Monte Carlo VaR, generated dynamically per run rather than a stored result
- **Phase 7** Expected Shortfall, and a VaR-methods comparison
- **Phase 8** GARCH(1,1) conditional volatility, re-estimated on every refresh, with residual
  diagnostics and a forecast-vs-realised comparison
- **Phase 9** Fundamental risk, reusing Module 1's ratios rather than recomputing them
- **Phase 10** Valuation risk and stress testing, reusing Module 2's DCF and WACC, including
  reverse stress testing against the current market price
- **Phase 11** Portfolio risk: correlation, risk contribution, concentration, Sharpe/Sortino
- **Phase 12** Backtesting (Kupiec, Christoffersen) and model validation
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
