"""Phase 3b: drawdown.

    Drawdown_t = (Value_t - Running Peak_t) / Running Peak_t

Always zero or negative: the running peak is the highest value seen up to and including
today, so today's value can never exceed it. A drawdown of -28% means the asset is 28% below
its own best-ever level as of that date, which is a different and more useful question than
"how volatile has this been," since two assets with identical volatility can have very
different drawdown experiences depending on whether their bad days cluster together.

**Duration and recovery are counted in trading days, not calendar days**, so a drawdown that
spans a holiday closure is not double-counted, and the two are kept as separate figures on
purpose: a drawdown can be deep but brief (a fast crash and V-shaped recovery) or shallow but
prolonged (a long grind), and collapsing both into a single "drawdown score" would hide which
one actually happened.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownEpisode:
    """One peak-to-trough-to-recovery cycle."""

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    trough_value_pct: float  # drawdown at the trough, e.g. -0.284
    duration_to_trough_days: int  # trading days from peak to trough
    recovery_date: pd.Timestamp | None  # None if not yet recovered as of the series end
    recovery_duration_days: int | None  # trading days from trough back to the prior peak


@dataclass(frozen=True)
class DrawdownSummary:
    current_drawdown: float
    max_drawdown: float
    max_drawdown_date: pd.Timestamp
    current_duration_days: int  # trading days since the running peak, 0 if at a new high
    is_in_drawdown: bool


def drawdown_series(prices: pd.Series) -> pd.Series:
    """The drawdown at every point in time, from the running peak up to that point."""
    running_peak = prices.cummax()
    return (prices - running_peak) / running_peak


def drawdown_summary(prices: pd.Series) -> DrawdownSummary:
    """The headline drawdown figures as of the last observation."""
    dd = drawdown_series(prices)
    running_peak = prices.cummax()

    current = float(dd.iloc[-1])
    max_dd = float(dd.min())
    max_dd_date = dd.idxmin()

    # Trading days since the most recent time the series was at its own running peak.
    at_peak = prices >= running_peak
    if at_peak.iloc[-1]:
        duration = 0
    else:
        last_peak_idx = at_peak[at_peak].index[-1] if at_peak.any() else prices.index[0]
        duration = int((prices.index >= last_peak_idx).sum()) - 1

    return DrawdownSummary(
        current_drawdown=current, max_drawdown=max_dd, max_drawdown_date=max_dd_date,
        current_duration_days=duration, is_in_drawdown=current < -1e-9,
    )


def identify_drawdown_episodes(prices: pd.Series, min_depth: float = 0.10) -> list[DrawdownEpisode]:
    """Every peak-to-trough-to-recovery cycle deeper than `min_depth`.

    A shallow, routine pullback is not a "drawdown episode" in the sense an analyst means
    when asking about a stock's crash history, which is why episodes below the threshold are
    not reported: reporting every 3% wiggle as an episode would bury the ones that matter.
    """
    running_peak = prices.cummax()
    dd = (prices - running_peak) / running_peak

    episodes: list[DrawdownEpisode] = []
    in_episode = False
    peak_date = peak_value = trough_date = trough_value = None

    for i, (date, value) in enumerate(prices.items()):
        if value >= running_peak.loc[date] - 1e-12:  # at a new high (or tied)
            if in_episode and trough_value is not None and (trough_value - peak_value) / peak_value <= -min_depth:
                episodes.append(DrawdownEpisode(
                    peak_date=peak_date, trough_date=trough_date,
                    trough_value_pct=(trough_value - peak_value) / peak_value,
                    duration_to_trough_days=int((prices.index.get_loc(trough_date) - prices.index.get_loc(peak_date))),
                    recovery_date=date,
                    recovery_duration_days=int(prices.index.get_loc(date) - prices.index.get_loc(trough_date)),
                ))
            in_episode, peak_date, peak_value = False, date, value
            trough_date, trough_value = None, None
        else:
            if not in_episode:
                in_episode = True
            if trough_value is None or value < trough_value:
                trough_date, trough_value = date, value

    # A drawdown still open at the end of the series has no recovery date yet.
    if in_episode and trough_value is not None and peak_value and (trough_value - peak_value) / peak_value <= -min_depth:
        episodes.append(DrawdownEpisode(
            peak_date=peak_date, trough_date=trough_date,
            trough_value_pct=(trough_value - peak_value) / peak_value,
            duration_to_trough_days=int(prices.index.get_loc(trough_date) - prices.index.get_loc(peak_date)),
            recovery_date=None, recovery_duration_days=None,
        ))

    return sorted(episodes, key=lambda e: e.trough_value_pct)
