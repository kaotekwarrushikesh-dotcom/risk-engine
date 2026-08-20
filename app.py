"""Live Institutional Risk Engine, as an app.

Type a ticker, get its market risk: volatility and regime, beta, drawdown, Value at Risk by
three methods, Expected Shortfall, a fitted GARCH(1,1) with conditional VaR, and a formal
backtest of whether any of it is calibrated.

The app is built around one editorial rule that the engine's own findings force. **No risk
number is shown on its own when the methods behind it disagree.** VaR gets all three methods
side by side because they disagree in a structured way, and Expected Shortfall sits next to
VaR because a threshold says nothing about what lies beyond it. A dashboard that picks one
number and prints it large is the failure mode this whole module was built to avoid.

Every figure carries its freshness. Data is classified DELAYED or STALE and never LIVE,
because the provider offers no real-time guarantee, and claiming otherwise would be the one
thing in a risk tool that is genuinely unsafe to get wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from risk_engine.backtesting import backtest_var, basel_traffic_light, compare_backtests
from risk_engine.beta import compute_beta, rolling_beta
from risk_engine.data_loader import clean_prices, fetch_prices, validate_prices
from risk_engine.drawdown import drawdown_series, drawdown_summary, identify_drawdown_episodes
from risk_engine.expected_shortfall import (
    compare_methods,
    historical_expected_shortfall,
    parametric_expected_shortfall,
)
from risk_engine.garch import conditional_var, fit_garch, forecast_volatility
from risk_engine.returns import log_returns
from risk_engine.settings import BENCHMARKS, DEFAULT_BENCHMARK, ROLLING_WINDOWS
from risk_engine.var_historical import historical_var, rolling_historical_var
from risk_engine.var_monte_carlo import convergence_path, monte_carlo_var
from risk_engine.var_parametric import fit_distribution, normal_vs_historical_gap, parametric_var
from risk_engine.volatility import (
    historical_volatility,
    identify_volatility_spikes,
    rolling_volatility,
    volatility_regime,
)

st.set_page_config(page_title="Live Institutional Risk Engine", page_icon="⚠️", layout="wide")

GREEN, AMBER, RED, BLUE, GREY = "#1e7a4b", "#b8860b", "#a32020", "#2b6cb0", "#8a8a8a"
REGIME_COLOUR = {"elevated": RED, "normal": GREEN, "subdued": BLUE, "unknown": GREY}
ZONE_COLOUR = {"green": GREEN, "yellow": AMBER, "red": RED}


def pct(v, dp=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dp}%}"


def num(v, dp=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{dp}f}"


def chart(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark", showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=900, show_spinner=False)
def load(ticker: str, period: str):
    """Prices, validation report and freshness. Cached 15 minutes: the data is delayed
    anyway, so refetching on every widget interaction buys nothing and costs a round trip."""
    raw, status = fetch_prices(ticker, period=period)
    report = validate_prices(raw, ticker)
    frame = clean_prices(raw)
    return frame, report, status


@st.cache_data(ttl=900, show_spinner=False)
def garch_fit(ticker: str, period: str):
    """GARCH is the expensive call: a maximum-likelihood fit over thousands of observations,
    cached separately so switching tabs does not refit it."""
    frame, _, _ = load(ticker, period)
    return fit_garch(log_returns(frame["adj_close"]))


@st.cache_data(ttl=900, show_spinner=False)
def backtest_series(ticker: str, period: str, confidence: float):
    """Rolling VaR by both approaches, for the backtest. The slowest thing in the app by far,
    since the conditional series refits GARCH every 50 days across the whole history."""
    from risk_engine.garch import rolling_conditional_var

    frame, _, _ = load(ticker, period)
    returns = log_returns(frame["adj_close"])
    unconditional = rolling_historical_var(returns, window=500, confidence=confidence)
    conditional = rolling_conditional_var(returns, confidence=confidence,
                                          refit_every=50, min_window=500)
    return returns, unconditional, conditional


st.title("Live Institutional Risk Engine")
st.caption(
    "Volatility, beta, drawdown, Value at Risk by three independent methods, Expected "
    "Shortfall, GARCH(1,1) conditional volatility and a formal backtest. Every method's own "
    "limitations are shown next to its numbers rather than left in a footnote."
)

controls = st.columns([2.2, 1.6, 1, 1])
ticker = controls[0].text_input("Ticker", value="AAPL",
                                placeholder="AAPL, MSFT, ^GSPC, RELIANCE.NS, SHEL.L").strip()
benchmark_label = controls[1].selectbox("Benchmark for beta", list(BENCHMARKS),
                                        index=list(BENCHMARKS).index(DEFAULT_BENCHMARK))
period = controls[2].selectbox("History", ["1y", "3y", "5y", "10y"], index=3)
confidence = controls[3].selectbox("Confidence", [0.95, 0.99], index=1,
                                   format_func=lambda c: f"{c:.0%}")
benchmark = BENCHMARKS[benchmark_label]

if not ticker:
    st.info("Enter a ticker to begin. Indices work too: `^GSPC`, `^NSEI`, `^FTSE`.")
    st.stop()

with st.spinner(f"Fetching and analysing {ticker}..."):
    try:
        frame, report, status = load(ticker, period)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load {ticker}: {exc}")
        st.stop()

if report.blocking:
    st.error("**Insufficient data for reliable risk calculation.** "
             + " ".join(report.blocking))
    st.stop()

prices = frame["adj_close"]
returns = log_returns(prices)

# --- Freshness, stated rather than implied ---------------------------------------------------

classification = status.classification
badge = {"DELAYED": AMBER, "STALE": GREY}.get(classification, GREY)
header = st.columns([1.4, 1, 1, 1, 1.2])
header[0].markdown(f"### {ticker}")
header[1].metric("Last close", num(float(prices.iloc[-1])))
header[2].metric("Observations", f"{len(frame):,}")
header[3].metric("History", f"{frame.index[0].date()} to {frame.index[-1].date()}")
header[4].markdown(
    f"<div style='padding-top:12px'><span style='background:{badge};color:#fff;"
    f"padding:3px 10px;border-radius:4px;font-size:0.8rem;font-weight:600'>"
    f"{classification}</span><br><span style='font-size:0.75rem;color:#999'>"
    f"fetched {status.fetched_at:%d %b %Y, %H:%M} · {status.source}"
    f"{' · cached' if status.from_cache else ''}</span></div>",
    unsafe_allow_html=True,
)

st.caption(
    "**Never LIVE.** The data provider offers no real-time service guarantee, so this engine "
    "classifies every fetch as DELAYED or STALE by measured age rather than claiming more "
    "than the source supports."
)

if report.issues_found:
    with st.expander(f"{len(report.issues_found)} data issue(s) corrected automatically"):
        for issue in report.issues_found:
            st.write(f"- {issue}")
if report.warnings:
    with st.expander(f"{len(report.warnings)} data warning(s)"):
        for warning in report.warnings:
            st.write(f"- {warning}")

st.divider()

tabs = st.tabs(["Overview", "Volatility", "Beta & drawdown", "Value at Risk",
                "Expected Shortfall", "GARCH", "Backtest", "Methodology"])

# =============================== OVERVIEW =====================================================

with tabs[0]:
    vol = historical_volatility(returns)
    rolling_vol = rolling_volatility(returns, window=60).dropna()
    current_vol = float(rolling_vol.iloc[-1]) if len(rolling_vol) else float("nan")
    regime = volatility_regime(current_vol, float(rolling_vol.mean()) if len(rolling_vol) else np.nan)
    dd = drawdown_summary(prices)
    hist_var = historical_var(returns, confidence)
    es = historical_expected_shortfall(returns, confidence)

    row = st.columns(5)
    row[0].metric("Annualised volatility", pct(vol.annualised, 1))
    row[1].metric("Current (60d)", pct(current_vol, 1), delta=regime.capitalize(),
                  delta_color="off")
    row[2].metric(f"{confidence:.0%} 1-day VaR", pct(hist_var.var_return))
    row[3].metric(f"{confidence:.0%} Expected Shortfall", pct(es.es_return))
    row[4].metric("Max drawdown", pct(dd.max_drawdown, 1))

    st.markdown(
        f"<span style='color:{REGIME_COLOUR.get(regime, GREY)};font-weight:600'>"
        f"Volatility regime: {regime}</span> — measured against this security's own history "
        "rather than an absolute level, since 'high volatility' means something different for "
        "a utility than for a speculative small cap.",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Adjusted close",
                             line=dict(color=BLUE, width=1.4)))
    chart(fig, 300)

    st.info(
        f"**Read these together, not one at a time.** The {confidence:.0%} VaR of "
        f"{pct(hist_var.var_return)} is a threshold: it says a worse day happens "
        f"{1 - confidence:.0%} of the time and says nothing about how much worse. Expected "
        f"Shortfall answers that, and here the average loss beyond the threshold is "
        f"{pct(es.es_return)}, {es.tail_severity:.2f} times the VaR itself. Both are "
        "unconditional; the GARCH tab has the figure conditioned on today's volatility."
    )

# =============================== VOLATILITY ====================================================

with tabs[1]:
    window = st.select_slider("Rolling window (trading days)", options=ROLLING_WINDOWS, value=60)
    rolling_vol = rolling_volatility(returns, window=window).dropna()

    cols = st.columns(4)
    cols[0].metric("Daily", pct(vol.daily, 2))
    cols[1].metric("Weekly", pct(vol.weekly, 2))
    cols[2].metric("Monthly", pct(vol.monthly, 2))
    cols[3].metric("Annualised", pct(vol.annualised, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                             name=f"{window}-day rolling, annualised",
                             line=dict(color=BLUE, width=1.3)))
    median = float(rolling_vol.median())
    fig.add_hline(y=median, line=dict(color=GREY, dash="dot"),
                  annotation_text=f"median {median:.1%}")
    fig.add_hline(y=median * 1.75, line=dict(color=RED, dash="dot"),
                  annotation_text="spike threshold (1.75x median)")
    chart(fig)

    spikes = identify_volatility_spikes(rolling_vol)
    if spikes.empty:
        st.caption("No sustained volatility spikes above 1.75x the series median in this window.")
    else:
        st.markdown("**Elevated-volatility episodes**, grouped into contiguous periods rather "
                    "than listed day by day, since a sustained stretch of stress reads as one "
                    "event to an analyst and not as forty.")
        display = spikes.copy()
        display["peak_volatility"] = display["peak_volatility"].map(lambda v: f"{v:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption(
        "The window is a modelling choice, not a display setting: a 20-day window reacts fast "
        "and is noisy, a 252-day window is stable and slow to reflect a regime change, and "
        "both are legitimate depending on what the number is for."
    )

# =============================== BETA & DRAWDOWN ================================================

with tabs[2]:
    left, right = st.columns(2)

    with left:
        st.subheader(f"Beta vs {benchmark_label}")
        try:
            benchmark_frame, _, _ = load(benchmark, period)
            benchmark_returns = log_returns(benchmark_frame["adj_close"])
            beta = compute_beta(returns, benchmark_returns)

            b = st.columns(3)
            b[0].metric("Beta", f"{beta.beta_regression:.3f}")
            b[1].metric("R-squared", f"{beta.r_squared:.3f}")
            b[2].metric("Confidence", beta.confidence.capitalize())

            fig = go.Figure()
            aligned_a, aligned_b = returns.align(benchmark_returns, join="inner")
            fig.add_trace(go.Scatter(x=aligned_b.values, y=aligned_a.values, mode="markers",
                                     name="Daily returns",
                                     marker=dict(size=3, color=BLUE, opacity=0.45)))
            xs = np.linspace(float(aligned_b.min()), float(aligned_b.max()), 50)
            fig.add_trace(go.Scatter(x=xs, y=beta.alpha + beta.beta_regression * xs,
                                     name="Fitted", line=dict(color=RED, width=2)))
            fig.update_layout(xaxis_title=benchmark_label, yaxis_title=ticker)
            chart(fig, 300)

            if beta.confidence == "low":
                st.warning(
                    f"R-squared is {beta.r_squared:.3f}, so this benchmark explains very little "
                    "of this security's movement. The beta is a number that looks meaningful "
                    "and largely is not; it is reported as low-confidence rather than hidden, "
                    "which is what the confidence field is for."
                )
            st.caption(
                "Beta is computed two independent ways from the same aligned data (the "
                "covariance formula and OLS) and the two are asserted equal at runtime. They "
                "are the same estimator by construction, so any gap between them is an "
                "alignment bug rather than a modelling choice."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Beta unavailable against {benchmark_label}: {exc}")

    with right:
        st.subheader("Drawdown")
        dd_series = drawdown_series(prices)
        summary = drawdown_summary(prices)

        d = st.columns(3)
        d[0].metric("Current", pct(summary.current_drawdown, 1))
        d[1].metric("Maximum", pct(summary.max_drawdown, 1))
        d[2].metric("Worst on", str(summary.max_drawdown_date.date()))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, name="Drawdown",
                                 fill="tozeroy", line=dict(color=RED, width=1)))
        chart(fig, 300)

        episodes = identify_drawdown_episodes(prices, min_depth=0.10)
        if episodes:
            st.markdown("**Episodes deeper than 10%**")
            st.dataframe(pd.DataFrame([
                {"Peak": str(e.peak_date.date()), "Trough": str(e.trough_date.date()),
                 "Depth": f"{e.trough_value_pct:.1%}",
                 "Recovered": str(e.recovery_date.date()) if e.recovery_date else "not yet"}
                for e in episodes
            ]), use_container_width=True, hide_index=True)
        st.caption(
            "Durations are in trading days, which understates elapsed calendar time across "
            "holidays. That is deliberate (trading days are what a rolling calculation "
            "consumes) but it means a duration should not be read as days out of the market."
        )

# =============================== VALUE AT RISK ===================================================

with tabs[3]:
    horizon = st.select_slider("Horizon (trading days)", options=[1, 5, 10, 20], value=1)
    portfolio_value = st.number_input("Position size (optional, to express VaR in currency)",
                                      min_value=0.0, value=0.0, step=100_000.0)
    value = portfolio_value if portfolio_value > 0 else None

    hist = historical_var(returns, confidence, horizon, value)
    normal = parametric_var(returns, confidence, horizon, value, "normal")
    student = parametric_var(returns, confidence, horizon, value, "t")
    simulated = monte_carlo_var(returns, confidence, horizon, value, paths=20_000,
                                draw_method="bootstrap")
    fit = fit_distribution(returns)

    st.markdown(f"#### {confidence:.0%} VaR over {horizon} day{'s' if horizon > 1 else ''}")
    v = st.columns(4)
    v[0].metric("Historical", pct(hist.var_return))
    v[1].metric("Parametric (normal)", pct(normal.var_return))
    v[2].metric("Parametric (Student-t)", pct(student.var_return))
    v[3].metric("Monte Carlo", pct(simulated.var_return),
                delta=f"±{simulated.standard_error:.2%}", delta_color="off")

    if value:
        st.caption(f"On a position of {value:,.0f}: historical VaR is "
                   f"{hist.var_value:,.0f}.")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=(np.exp(returns) - 1).values, nbinsx=120, name="Daily returns",
                               marker=dict(color=BLUE, opacity=0.55)))
    for label, v_return, colour in [("Historical", hist.var_return, RED),
                                    ("Normal", normal.var_return, AMBER),
                                    ("Student-t", student.var_return, GREEN)]:
        fig.add_vline(x=-v_return, line=dict(color=colour, dash="dash"),
                      annotation_text=label, annotation_position="top")
    fig.update_layout(xaxis_title="Daily return", yaxis_title="Days")
    chart(fig)

    gap = normal_vs_historical_gap(normal.var_return, hist.var_return)
    st.info(f"**Normal vs empirical:** {gap['reading']}")

    if not fit.is_normal:
        st.warning(
            f"**The normal assumption is rejected here** (Jarque-Bera p = "
            f"{fit.jarque_bera_p:.2g}, excess kurtosis {fit.excess_kurtosis:.2f} against 0 for "
            "a normal). It is not simply optimistic though: fitting a normal to a fat-tailed "
            "sample inflates sigma to absorb the extreme days, which tends to push the 95% "
            "figure too far out while still falling short at 99%. Which way it errs depends on "
            "where you read it."
        )

    for warning in hist.warnings:
        st.warning(warning)

    st.caption(
        f"Estimated from {hist.n_observations:,} observations, of which "
        f"{hist.tail_observations:.0f} lie beyond the quantile. Bootstrap interval for the "
        f"historical estimate: {pct(hist.ci_low)} to {pct(hist.ci_high)}. Worst single loss in "
        f"the sample: {pct(hist.worst_observed_loss)}."
    )

    with st.expander("Monte Carlo convergence: is the path count enough?"):
        conv = convergence_path(returns, confidence)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=conv["paths"], y=conv["var_return"], mode="lines+markers",
                                 name="VaR estimate", line=dict(color=BLUE)))
        fig.update_layout(xaxis_title="Simulated paths", yaxis_title="VaR", xaxis_type="log")
        chart(fig, 280)
        st.caption(
            "If the estimate is still moving at 10,000 paths, the number being quoted is "
            "partly simulation noise. A far-tail quantile on fat-tailed data converges slowly."
        )

# =============================== EXPECTED SHORTFALL ================================================

with tabs[4]:
    st.markdown(f"#### {confidence:.0%} Expected Shortfall")
    st.caption(
        "VaR is a threshold and says nothing about what lies beyond it. Two portfolios can "
        "share an identical VaR while one loses a further 1% in the tail and the other loses "
        "everything. Expected Shortfall averages the losses past the threshold instead."
    )

    table = compare_methods(returns, confidence, paths=20_000)
    display = table.copy()
    display["var"] = display["var"].map(lambda v: f"{v:.2%}")
    display["es"] = display["es"].map(lambda v: f"{v:.2%}")
    display["tail_severity"] = display["tail_severity"].map(lambda v: f"{v:.2f}x")
    display["var_vs_historical"] = display["var_vs_historical"].map(lambda v: f"{v:+.2%}")
    display.columns = ["Method", "Assumption", "VaR", "ES", "ES / VaR", "VaR vs historical"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    empirical = historical_expected_shortfall(returns, confidence)
    normal_es = parametric_expected_shortfall(returns, confidence, distribution="normal")
    shortfall = (empirical.es_return - normal_es.es_return) / empirical.es_return

    st.info(
        f"**The normal model fails worse on ES than on VaR.** Its Expected Shortfall of "
        f"{pct(normal_es.es_return)} is {shortfall:.0%} below the empirical "
        f"{pct(empirical.es_return)}. A normal distribution has a fixed thin tail, so its "
        f"ES/VaR ratio is pinned near 1.15 regardless of the data, while the real ratio here "
        f"is {empirical.tail_severity:.2f}. The error compounds precisely when you ask the "
        "question ES exists to answer."
    )
    st.caption(
        "Expected Shortfall is coherent and VaR is not: VaR can report that diversification "
        "increased risk, which is a defect of the measure rather than a fact about the "
        "positions. That is why Basel moved market-risk capital from 99% VaR to 97.5% ES."
    )
    for warning in empirical.warnings:
        st.warning(warning)

# =============================== GARCH ============================================================

with tabs[5]:
    st.markdown("#### GARCH(1,1): risk given today, not risk on average")
    with st.spinner("Fitting GARCH..."):
        try:
            fit_g = garch_fit(ticker, period)
        except Exception as exc:  # noqa: BLE001
            st.error(f"GARCH could not be fitted: {exc}")
            st.stop()

    conditional = conditional_var(fit_g, confidence, horizon_days=1)
    unconditional = historical_var(returns, confidence)

    g = st.columns(5)
    g[0].metric("Current volatility", pct(fit_g.current_annualised_volatility, 1))
    g[1].metric("Realised (sample)", pct(fit_g.realised_annualised_volatility, 1))
    g[2].metric("Regime", conditional["regime"].capitalize())
    g[3].metric(f"Conditional {confidence:.0%} VaR", pct(conditional["var_return"]))
    g[4].metric("Unconditional", pct(unconditional.var_return),
                delta=f"{conditional['var_return'] - unconditional.var_return:+.2%}",
                delta_color="off")

    p = st.columns(4)
    p[0].metric("alpha (reaction)", f"{fit_g.alpha:.3f}")
    p[1].metric("beta (persistence)", f"{fit_g.beta:.3f}")
    p[2].metric("alpha + beta", f"{fit_g.persistence:.4f}")
    p[3].metric("Half-life", f"{fit_g.half_life_days:.0f} days")

    cond_vol = fit_g.conditional_volatility / 100.0 * np.sqrt(252)
    forecast = forecast_volatility(fit_g, 30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol.values, name="Conditional volatility",
                             line=dict(color=BLUE, width=1.2)))
    future = pd.date_range(cond_vol.index[-1], periods=31, freq="B")[1:]
    fig.add_trace(go.Scatter(x=future, y=forecast.annualised_volatility.values,
                             name="30-day forecast", line=dict(color=AMBER, width=2, dash="dash")))
    fig.add_hline(y=fit_g.realised_annualised_volatility, line=dict(color=GREY, dash="dot"),
                  annotation_text="realised sample volatility")
    chart(fig)

    st.info(
        "**This is the only VaR here that moves with the regime.** Every other figure in this "
        "app is unconditional: it describes the return distribution across the whole sample, "
        "blending calm periods with crises. The forecast mean-reverts toward the long-run "
        "level rather than projecting today's volatility unchanged, which is exactly what "
        "square-root-of-time scaling cannot do."
    )

    if fit_g.residuals_are_clean:
        st.success(
            f"Standardised residuals show no remaining ARCH effects (LM p = "
            f"{fit_g.arch_test_after.p_value:.3f}), so the model absorbed the clustering it "
            "was fitted to capture. Before fitting, the same test gave p = "
            f"{fit_g.arch_test_before.p_value:.2g}."
        )
    for warning in fit_g.warnings:
        st.warning(warning)

    if not fit_g.long_run_is_reliable:
        st.caption(
            "The model-implied long-run volatility is suppressed above 0.995 persistence "
            "because omega / (1 - persistence) divides by a number approaching zero and "
            "becomes numerically explosive. Realised volatility is used as the benchmark "
            "instead, being a measurement rather than an extrapolation."
        )

# =============================== BACKTEST ==========================================================

with tabs[6]:
    st.markdown("#### Is any of this actually calibrated?")
    st.caption(
        "Without a backtest, 'GARCH is the better model' is an appeal to authority. Kupiec "
        "tests whether the breach *count* is right; Christoffersen tests whether breaches "
        "*cluster*. A model can have exactly the right number of breaches and still be badly "
        "wrong if they all land in the same fortnight."
    )

    if len(returns) < 700:
        st.info(
            f"Only {len(returns):,} observations. The backtest needs a 500-day warm-up before "
            "it can start scoring, so a longer history is required: select 5y or 10y above."
        )
    elif st.button("Run backtest (slow: refits GARCH across the whole history)"):
        with st.spinner("Backtesting..."):
            rets, uncond, cond = backtest_series(ticker, period, confidence)
            table = compare_backtests(rets, {"Historical (unconditional)": uncond,
                                             "GARCH (conditional)": cond}, confidence)

        display = table.copy()
        display["rate"] = display["rate"].map(lambda v: f"{v:.2%}")
        for column in ("kupiec_p", "christoffersen_p", "joint_p"):
            display[column] = display[column].map(lambda v: f"{v:.4f}")
        display.columns = ["Model", "Breaches", "Expected", "Rate", "Kupiec p",
                           "Christoffersen p", "Joint p", "Coverage", "Independence", "Max run"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        fig = go.Figure()
        realised = -(np.exp(rets) - 1.0)
        fig.add_trace(go.Scatter(x=realised.index, y=realised.values, name="Realised loss",
                                 line=dict(color=GREY, width=0.6)))
        fig.add_trace(go.Scatter(x=uncond.index, y=uncond.values, name="Historical VaR",
                                 line=dict(color=AMBER, width=1.4)))
        fig.add_trace(go.Scatter(x=cond.index, y=cond.values, name="GARCH conditional VaR",
                                 line=dict(color=BLUE, width=1.4)))
        chart(fig, 380)

        # The traffic light is defined on the most recent 250 trading days, so it is scored on
        # exactly that window. Counting breaches across the whole backtest and comparing them
        # against a 250-day threshold would force RED on any well-calibrated model with a long
        # history, which says nothing about the model and everything about the arithmetic.
        scored = cond.dropna().index[-250:]
        recent = backtest_var(rets.loc[scored], cond.loc[scored], confidence)
        zone = basel_traffic_light(recent.n_breaches, len(scored))
        st.markdown(
            f"**Basel traffic light (GARCH), most recent {len(scored)} trading days:** "
            f"<span style='background:{ZONE_COLOUR[zone['zone']]};color:#fff;padding:2px 10px;"
            f"border-radius:4px'>{zone['zone'].upper()}</span> — {zone['meaning']} "
            f"({recent.n_breaches} breaches against a green ceiling of {zone['green_max']:.0f}).",
            unsafe_allow_html=True,
        )
        st.caption(
            "The likelihood-ratio tests above score the full backtest window; the traffic "
            "light scores only the last 250 days, because that is how the supervisory "
            "framework is defined. They can disagree, and when they do the tests are the more "
            "informative of the two."
        )
        full = backtest_var(rets, cond, confidence)
        for warning in full.warnings:
            st.warning(warning)
        st.caption(
            "Both series use only data available up to each date. A VaR series fitted on the "
            "whole sample would know about future crises and would pass any backtest "
            "trivially, which is precisely the error a backtest exists to catch."
        )
    else:
        st.caption("The backtest refits GARCH every 50 days across the full history, so it is "
                   "run on demand rather than on every page load.")

# =============================== METHODOLOGY =========================================================

with tabs[7]:
    st.markdown("""
#### What this engine will and will not claim

**"Live" means refreshable on demand, not real-time.** The data provider is a free,
unofficial interface with no service-level agreement, so every fetch is classified DELAYED or
STALE by measured age and never LIVE, regardless of how fresh a given pull happens to be.

**Validation is tiered rather than pass/fail.** Issues are corrected automatically (a
duplicate date, an out-of-order row), warnings let the calculation proceed but flag what to
read carefully, and blocking issues refuse the series outright rather than returning a
confident wrong number. The cleaning step applies exactly the corrections the validator
describes, so the two can never quietly disagree.

**No method is shown alone where the methods disagree.** VaR appears three ways because they
diverge in a structured way, and Expected Shortfall sits beside it because a threshold says
nothing about what lies past it.

#### Each method's own limitations

| Method | What it cannot do |
|---|---|
| Historical VaR | Cannot produce a loss worse than one already observed, so a quiet sample gives a reassuring number for a structural reason rather than an empirical one |
| Parametric (normal) | Assumes a distribution equity returns reject; errs in opposite directions at 95% and 99% |
| Parametric (Student-t) | Models the tail's shape better than its threshold: tends to understate VaR while getting ES about right |
| Monte Carlo (bootstrap) | Shares historical VaR's one-day ceiling; only multi-day paths escape it |
| All except GARCH | Unconditional: describes the whole sample, not today's regime |
| Multi-day figures | Assume independent days, so a real stressed fortnight is worse than any of them implies |
| GARCH here | Symmetric, so it does not let downside shocks raise volatility more than upside ones. On the Nifty this over-breaches badly enough to fail coverage; GJR-GARCH is the untested fix |

#### Backtest caveat

At 250 observations these tests have low power, so "not rejected" is a much weaker statement
than it appears and a mediocre model routinely survives. Sample size travels with every
result for that reason.
""")
