"""Configuration for the risk engine: benchmarks, cache location, default parameters.

Kept as plain constants rather than a config file format (YAML/TOML), since every value
here is read by exactly one or two call sites and a config file would add an indirection
layer without adding flexibility a Python module doesn't already give for free.
"""

from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "risk_engine"

# Benchmark indices a user can select, verified live against yfinance before being listed
# here. A benchmark that silently returns nothing would make beta and VaR-vs-market
# comparisons wrong without an error, so nothing goes in this list unmeasured.
BENCHMARKS: dict[str, str] = {
    "S&P 500 (US)": "^GSPC",
    "STOXX Europe 600": "^STOXX",
    "FTSE 100 (UK)": "^FTSE",
    "ISEQ (Ireland)": "^ISEQ",
    "Nifty 50 (India)": "^NSEI",
    "DAX (Germany)": "^GDAXI",
}
DEFAULT_BENCHMARK = "S&P 500 (US)"

# Rolling windows offered for volatility and beta, in trading days.
ROLLING_WINDOWS = [20, 60, 120, 252]
DEFAULT_ROLLING_WINDOW = 60

CONFIDENCE_LEVELS = [0.95, 0.99]
DEFAULT_CONFIDENCE = 0.95

VAR_HORIZONS_DAYS = [1, 10]
DEFAULT_VAR_HORIZON = 1

MONTE_CARLO_PATHS = [10_000, 50_000, 100_000]
DEFAULT_MONTE_CARLO_PATHS = 10_000

TRADING_DAYS_PER_YEAR = 252

# Below this many return observations, results are not reported: a VaR or volatility
# figure from a handful of days is not wrong so much as meaningless, and the difference
# matters for how a failure is communicated.
MIN_OBSERVATIONS = 30

# Data is treated as delayed rather than live above this age, and stale beyond it. yfinance
# is an unofficial, free interface with no real-time guarantee, so "live" is never claimed
# for it regardless of how fresh a given pull happens to be.
DELAYED_AFTER_MINUTES = 15
STALE_AFTER_HOURS = 24
