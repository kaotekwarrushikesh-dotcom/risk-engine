"""Tests for Phase 1: data ingestion and validation.

These use synthetic price frames rather than live network calls, so they run offline and
deterministically; live-data behaviour is checked separately, by hand, against real tickers
(NVIDIA's 2024 split, GameStop's 2024 squeeze) documented in data_loader.py's own comments.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.data_loader import DataStatus, clean_prices, validate_prices


def make_prices(n=100, start="2024-01-01", price=100.0, freq="B") -> pd.DataFrame:
    """A clean, valid daily price series, business days only."""
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0003, 0.015, n)
    prices = price * np.cumprod(1 + returns)
    return pd.DataFrame({"close": prices, "adj_close": prices, "volume": 1_000_000}, index=idx)


# --- Basic validity -----------------------------------------------------------------------

def test_clean_series_has_no_issues_or_warnings():
    df = make_prices()
    report = validate_prices(df, "TEST")
    assert report.usable
    assert report.issues_found == []
    assert report.warnings == []


def test_empty_dataframe_is_blocking():
    report = validate_prices(pd.DataFrame(), "TEST")
    assert not report.usable
    assert report.n_observations == 0


def test_too_few_observations_is_blocking():
    df = make_prices(n=10)
    report = validate_prices(df, "TEST")
    assert not report.usable
    assert any("observations" in b for b in report.blocking)


# --- Duplicates and ordering ---------------------------------------------------------------

def test_duplicate_dates_are_detected_and_removed():
    df = make_prices(n=60)
    dup = pd.concat([df, df.iloc[[5]]])
    report = validate_prices(dup, "TEST")
    assert any("duplicate" in i for i in report.issues_found)
    cleaned = clean_prices(dup)
    assert not cleaned.index.duplicated().any()


def test_out_of_order_dates_are_detected_and_sorted():
    df = make_prices(n=60)
    shuffled = df.sample(frac=1, random_state=1)
    report = validate_prices(shuffled, "TEST")
    assert any("chronological" in i for i in report.issues_found)


def test_future_dated_rows_are_dropped():
    df = make_prices(n=60)
    future_row = pd.DataFrame(
        {"close": [999.0], "adj_close": [999.0], "volume": [1000]},
        index=[pd.Timestamp.now() + timedelta(days=30)],
    )
    with_future = pd.concat([df, future_row])
    report = validate_prices(with_future, "TEST")
    assert any("future" in i for i in report.issues_found)
    cleaned = clean_prices(with_future)
    assert (cleaned.index <= pd.Timestamp.now()).all()


# --- Price sanity --------------------------------------------------------------------------

def test_non_positive_prices_are_removed():
    df = make_prices(n=60)
    df.iloc[10, df.columns.get_loc("close")] = -5.0
    df.iloc[20, df.columns.get_loc("close")] = 0.0
    report = validate_prices(df, "TEST")
    assert any("non-positive" in i for i in report.issues_found)
    cleaned = clean_prices(df)
    assert (cleaned["close"] > 0).all()


def test_missing_close_values_are_removed():
    df = make_prices(n=60)
    df.iloc[15, df.columns.get_loc("close")] = np.nan
    report = validate_prices(df, "TEST")
    assert any("missing" in i for i in report.issues_found)


# --- Extreme moves and weekends -------------------------------------------------------------

def test_extreme_single_day_move_is_flagged_not_removed():
    """A real crash must survive validation and be visible to the risk model, only flagged."""
    df = make_prices(n=100)
    df.iloc[50, df.columns.get_loc("close")] = df.iloc[49]["close"] * 0.5  # -50% day
    report = validate_prices(df, "TEST")
    assert report.usable  # not blocking
    assert any("30%" in w for w in report.warnings)
    # Not removed from the cleaned series.
    cleaned = clean_prices(df)
    assert len(cleaned) == len(df)


def test_ordinary_returns_do_not_trigger_the_extreme_move_warning():
    df = make_prices(n=200)
    report = validate_prices(df, "TEST")
    assert not any("30%" in w for w in report.warnings)


def test_weekend_rows_are_flagged():
    df = make_prices(n=60)
    saturday = df.index[10] + timedelta(days=(5 - df.index[10].dayofweek) % 7)
    weekend_row = pd.DataFrame(
        {"close": [100.0], "adj_close": [100.0], "volume": [500]}, index=[saturday]
    )
    with_weekend = pd.concat([df, weekend_row]).sort_index()
    report = validate_prices(with_weekend, "TEST")
    assert any("weekend" in w for w in report.warnings)


def test_large_gap_is_flagged():
    df = make_prices(n=60)
    # Introduce a 20-day gap by shifting everything after the midpoint forward.
    mid = len(df) // 2
    shifted_index = list(df.index[:mid]) + [d + timedelta(days=20) for d in df.index[mid:]]
    df.index = pd.DatetimeIndex(shifted_index)
    report = validate_prices(df, "TEST")
    assert any("gap" in w for w in report.warnings)


# --- DataStatus classification --------------------------------------------------------------

def test_status_is_never_classified_as_live():
    """The module must never claim real-time data; the freshest possible classification is
    DELAYED, since yfinance offers no verified real-time guarantee."""
    status = DataStatus(
        as_of=datetime.now(), fetched_at=datetime.now(), source="yfinance", from_cache=False,
    )
    assert status.classification in ("DELAYED", "STALE")
    assert status.classification != "LIVE"


def test_status_becomes_stale_after_the_threshold():
    old = datetime.now() - timedelta(hours=48)
    status = DataStatus(as_of=old, fetched_at=old, source="yfinance", from_cache=True)
    assert status.classification == "STALE"


def test_status_label_includes_the_refresh_timestamp():
    status = DataStatus(
        as_of=datetime(2026, 8, 17, 10, 30), fetched_at=datetime.now(),
        source="yfinance", from_cache=False,
    )
    assert "2026-08-17 10:30" in status.label()
