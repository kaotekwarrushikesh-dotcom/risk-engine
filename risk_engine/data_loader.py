"""Phase 1: data ingestion, caching, and validation.

    Data source -> ingestion -> validation -> (risk calculations, elsewhere)

Everything downstream of this module trusts that a DataFrame passed to it has already
cleared `validate_prices`. That is a deliberate boundary: risk calculations should not have
to re-check for duplicate dates or non-trading rows, and this module should not have to know
anything about VaR or volatility. If the data is not good enough to compute risk from, that
is decided once, here, and the caller is told plainly rather than getting a number back that
looks fine and isn't.

**"Live" is never claimed for what this module returns.** yfinance is a free, unofficial
interface to Yahoo Finance with no real-time guarantee or service level, so every fetch is
timestamped and classified as LIVE, DELAYED or STALE by how old the data actually is, not by
what the source claims. A dashboard built on this module shows that classification next to
every number it derives from price data, per the module's own requirement that delayed data
must never be presented as real-time.
"""

import io
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine.settings import (
    CACHE_DIR,
    DELAYED_AFTER_MINUTES,
    MIN_OBSERVATIONS,
    STALE_AFTER_HOURS,
)


@dataclass(frozen=True)
class DataStatus:
    """How fresh the data actually is, classified by measurement rather than by claim."""

    as_of: datetime
    fetched_at: datetime
    source: str
    from_cache: bool

    @property
    def age_minutes(self) -> float:
        return (datetime.now() - self.fetched_at).total_seconds() / 60.0

    @property
    def classification(self) -> str:
        if self.age_minutes <= DELAYED_AFTER_MINUTES:
            return "DELAYED"  # yfinance has no verified real-time SLA; never claim LIVE
        if self.age_minutes <= STALE_AFTER_HOURS * 60:
            return "DELAYED"
        return "STALE"

    def label(self) -> str:
        return (
            f"Last data refresh: {self.as_of:%Y-%m-%d %H:%M}  "
            f"(fetched {self.fetched_at:%Y-%m-%d %H:%M}, {self.classification})"
        )


@dataclass(frozen=True)
class DataQualityReport:
    """What was found while validating a price series, and whether it is usable."""

    ticker: str
    n_observations: int
    date_range: tuple[str, str] | None
    issues_found: list[str]
    warnings: list[str]
    blocking: list[str]

    @property
    def usable(self) -> bool:
        return not self.blocking


def _cache_path(ticker: str, period: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("^", "_").replace("/", "_")
    return CACHE_DIR / f"prices_{safe}_{period}.json"


def fetch_prices(
    ticker: str, period: str = "3y", refresh: bool = False,
) -> tuple[pd.DataFrame, DataStatus]:
    """Daily OHLCV for one ticker, cached to disk, with an honest freshness classification.

    `period` follows yfinance's own vocabulary ("1y", "3y", "5y", "max") rather than exact
    dates, so the cache key is stable across calls that mean the same thing.
    """
    cache = _cache_path(ticker, period)
    fetched_at = datetime.now()
    from_cache = False

    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        from_cache = True
        # pd.read_json treats a bare string ambiguously (path vs. content) depending on
        # version, so the content is wrapped in a StringIO to force it to be read as data.
        df = pd.read_json(io.StringIO(payload["data"]), orient="split")
        df.index = pd.to_datetime(df.index)
    else:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.Ticker(ticker).history(period=period, auto_adjust=False)

        if raw.empty:
            raise ValueError(f"{ticker}: no price data returned")

        df = pd.DataFrame({
            "close": raw["Close"],
            "adj_close": raw.get("Close") if "Adj Close" not in raw else raw["Adj Close"],
            "volume": raw["Volume"],
        })
        df.index = pd.to_datetime(df.index.tz_localize(None) if df.index.tz else df.index)
        cache.write_text(json.dumps({
            "fetched_at": fetched_at.isoformat(),
            "data": df.to_json(orient="split", date_format="iso"),
        }))
        time.sleep(0.1)

    status = DataStatus(
        as_of=df.index[-1].to_pydatetime(), fetched_at=fetched_at,
        source="yfinance", from_cache=from_cache,
    )
    return df, status


def validate_prices(df: pd.DataFrame, ticker: str) -> DataQualityReport:
    """Run every check before any risk figure is computed from this series.

    Checks are grouped by severity. A blocking issue means the series cannot support a
    reliable calculation at all (too few observations, no valid prices); a warning means the
    calculation can proceed but the result should be read with the warning in mind (a data
    gap, a suspected corporate action). The distinction matters because collapsing both into
    one undifferentiated "there were issues" message would make a user unable to tell
    "ignore this" from "do not trust this number."
    """
    issues: list[str] = []
    warns: list[str] = []
    blocking: list[str] = []

    if df.empty:
        return DataQualityReport(ticker, 0, None, ["no data returned"], [], ["no data returned"])

    # Duplicate observations
    dup_count = int(df.index.duplicated().sum())
    if dup_count:
        issues.append(f"{dup_count} duplicate date(s) found and removed")
        df = df[~df.index.duplicated(keep="last")]

    # Chronological order / incorrect dates
    if not df.index.is_monotonic_increasing:
        issues.append("dates were not in chronological order and were sorted")
        df = df.sort_index()

    future = df.index[df.index > pd.Timestamp.now()]
    if len(future):
        issues.append(f"{len(future)} observation(s) dated in the future, dropped")
        df = df[df.index <= pd.Timestamp.now()]

    # Non-trading days: a data provider occasionally includes a weekend row, usually a sign
    # of a bad timestamp rather than genuine trading activity.
    weekend_rows = int(df.index.dayofweek.isin([5, 6]).sum())
    if weekend_rows:
        warns.append(f"{weekend_rows} observation(s) fall on a weekend, which is unusual "
                     "for daily equity data and may indicate a timestamp issue")

    # Missing observations: gaps materially larger than a long weekend suggest missing data
    # rather than a holiday calendar this check does not know about.
    gaps = df.index.to_series().diff().dt.days.dropna()
    large_gaps = gaps[gaps > 7]
    if len(large_gaps):
        warns.append(
            f"{len(large_gaps)} gap(s) longer than 7 calendar days, the largest "
            f"{int(large_gaps.max())} days. This can be a trading halt, a delisting period, "
            "or missing data, and is not distinguished automatically."
        )

    # Non-positive or missing prices
    bad_price = int((df["close"] <= 0).sum() + df["close"].isna().sum())
    if bad_price:
        issues.append(f"{bad_price} non-positive or missing close price(s) removed")
        df = df[(df["close"] > 0) & df["close"].notna()]

    # Extreme single-day moves: flagged, not removed, since a genuine crash is a real
    # observation a risk model needs to see, not an error to clean away. Verified directly
    # against NVIDIA's June 2024 10:1 split: yfinance's `close` is already split-adjusted
    # regardless of the auto_adjust flag (only dividends create the close/adj_close gap), so
    # an extreme move here is a real price move or a genuine data anomaly, not an unadjusted
    # split, and the warning does not blame splits for something they no longer cause.
    if len(df) > 1:
        daily_return = df["close"].pct_change().dropna()
        extreme = daily_return[daily_return.abs() > 0.30]
        if len(extreme):
            dates = ", ".join(d.strftime("%Y-%m-%d") for d in extreme.index[:5])
            warns.append(
                f"{len(extreme)} single-day move(s) beyond 30% ({dates}"
                f"{'...' if len(extreme) > 5 else ''}). This is consistent with a genuine "
                "crash or spike, which a risk model should see rather than have smoothed "
                "away, but is also the signature of a bad print or a data-vendor error, so "
                "treat these dates with care rather than at face value."
            )

    # Inconsistent frequency: the typical gap should be close to 1 trading day; a series
    # mixing daily and weekly rows would silently corrupt every rolling-window calculation.
    if len(gaps):
        typical_gap = float(gaps.median())
        if typical_gap > 4:
            blocking.append(
                f"typical spacing between observations is {typical_gap:.0f} days, too "
                "sparse for a daily risk model"
            )

    n = len(df)
    if n < MIN_OBSERVATIONS:
        blocking.append(
            f"only {n} usable observations, below the {MIN_OBSERVATIONS} needed for a "
            "reliable risk calculation"
        )

    date_range = (df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")) if n else None

    return DataQualityReport(ticker, n, date_range, issues, warns, blocking)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same corrections `validate_prices` describes, so the two never disagree."""
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index <= pd.Timestamp.now()]
    df = df[(df["close"] > 0) & df["close"].notna()]
    return df


def load_and_validate(ticker: str, period: str = "3y", refresh: bool = False) -> tuple[pd.DataFrame, DataStatus, DataQualityReport]:
    """The full Phase 1 pipeline: fetch, validate, clean. Raises if the data is not usable."""
    raw, status = fetch_prices(ticker, period, refresh)
    report = validate_prices(raw, ticker)
    if not report.usable:
        raise ValueError(
            f"Insufficient data for reliable risk calculation on {ticker}: "
            + "; ".join(report.blocking)
        )
    return clean_prices(raw), status, report
