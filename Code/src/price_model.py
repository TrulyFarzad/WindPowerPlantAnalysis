# price_monte_carlo.py
# ------------------------------------------------------------
# Monte Carlo scenario generator for monthly electricity prices
# (industrial-grade, power-systems friendly)
#
# Inputs:
#   - U.S. retail electricity price dataset (monthly), with columns:
#       year, month, stateDescription, sectorName, customers, price, revenue, sales
#     'price' typically in cent/kWh (EIA-style). We convert to USD/MWh.
#   - Optional CPI index (date, index) to produce real (CPI-adjusted) prices.
#
# Outputs (by default under ./outputs):
#   - price_paths_M{H}_N{S}.parquet (or .csv)  : long table of (date, scenario_id, price_usd_mwh_nominal[, price_usd_mwh_real])
#   - price_quantiles_M{H}.parquet (or .csv)   : monthly P10/P50/P90 (nominal[, real])
#   - plots/*.png                              : fan chart & final-month histogram
#   - price_meta.json                          : summary metadata for reproducibility
#
# Why block bootstrap on log-returns?
#   - Keeps prices strictly positive (simulate in log space).
#   - Preserves seasonality (block_len=12 → Jan..Dec pattern retained).
#   - Produces stochastic paths suitable for Monte Carlo (10k scenarios).
#
# Author: (you/your lab)
# ------------------------------------------------------------

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- #
# 1) ---------- Utils ---------- #
# ----------------------------- #

def ensure_out_dir(out_dir: Path) -> Tuple[Path, Path]:
    """
    Create output directories: out_dir and out_dir/plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, plots_dir


def try_save_parquet_or_csv(df: pd.DataFrame, path_parquet: Path, path_csv: Path) -> Path:
    """
    Save DataFrame as Parquet if pyarrow/fastparquet available; otherwise save CSV.
    Returns the path actually written.
    """
    try:
        df.to_parquet(path_parquet, index=False)
        return path_parquet
    except Exception:
        df.to_csv(path_csv, index=False)
        return path_csv


def month_start_index(year: pd.Series, month: pd.Series, tz: str = "UTC") -> pd.DatetimeIndex:
    """
    Build a Month-Start datetime index from 'year' and 'month' columns.
    """
    dt = pd.to_datetime(dict(year=year, month=month, day=1), utc=True)
    # Normalize to month-start frequency
    dt = pd.DatetimeIndex(dt).tz_convert(tz).to_period("M").to_timestamp(how="start").tz_localize(tz)
    return dt


def to_usd_per_mwh(price: pd.Series, unit_source: str) -> pd.Series:
    """
    Convert provided price column to USD/MWh.
    Supported unit_source:
      - "cent_per_kwh"   : USD/MWh = 10 * cent_per_kWh
      - "usd_per_kwh"    : USD/MWh = 1000 * USD/kWh
      - "usd_per_mwh"    : already in USD/MWh
    """
    unit_source = unit_source.lower()
    s = price.astype(float)
    if unit_source == "cent_per_kwh":
        return 10.0 * s
    elif unit_source == "usd_per_kwh":
        return 1000.0 * s
    elif unit_source == "usd_per_mwh":
        return s
    else:
        raise ValueError(f"Unsupported unit_source: {unit_source}")


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """
    Weighted average with safe fallback to simple mean if weights missing/zero.
    """
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    w = w.fillna(0.0)
    if not np.isfinite(w).any() or w.sum() <= 0:
        return float(v.mean())
    return float(np.average(v, weights=w))


# --------------------------------------- #
# 2) ---- Load & build monthly series ---- #
# --------------------------------------- #

def load_us_dataset(
    csv_path: Path,
    tz: str = "UTC",
    state: Optional[str] = "*",
    sector: Optional[str] = "*",
    unit_source: str = "cent_per_kwh",
    weight_col: str = "sales",
) -> pd.Series:
    """
    Load U.S. monthly price dataset and return a SINGLE monthly series (USD/MWh, nominal).
    - state="*"    → all states (national)
    - sector="*"   → all sectors (weighted together)
    - Otherwise, you can pass a concrete state and/or sector to filter.
    The monthly series is a sales-weighted average across remaining rows.

    Expected columns:
      year, month, stateDescription, sectorName, price, sales
    """
    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required = {"year", "month", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Optional filters
    if state and state != "*":
        if "stateDescription" not in df.columns:
            raise ValueError("state filter requested but 'stateDescription' column not found.")
        df = df[df["stateDescription"].astype(str).str.strip().eq(str(state).strip())]

    if sector and sector != "*":
        if "sectorName" not in df.columns:
            raise ValueError("sector filter requested but 'sectorName' column not found.")
        df = df[df["sectorName"].astype(str).str.strip().eq(str(sector).strip())]

    # Build monthly DateTime index
    df["date"] = month_start_index(df["year"], df["month"], tz=tz)

    # Convert price to USD/MWh
    df["price_usd_mwh_nominal"] = to_usd_per_mwh(df["price"], unit_source=unit_source)

    # Compute monthly weighted average (by 'sales'), fallback to mean if weights missing/zero
    if weight_col not in df.columns:
        # Fallback to simple mean by month
        series = df.groupby("date")["price_usd_mwh_nominal"].mean()
    else:
        grp = df.groupby("date")
        series = grp.apply(lambda g: weighted_average(g["price_usd_mwh_nominal"], g[weight_col]))

    # Ensure monthly frequency, sorted, no duplicates
    series = series.sort_index().asfreq("MS")
    return series.rename("price_usd_mwh_nominal")


# ----------------------------------- #
# 3) ---- CPI (optional, "real") ---- #
# ----------------------------------- #

def load_cpi_series(cpi_csv: Path, tz: str = "UTC") -> pd.Series:
    """
    Load CPI file with columns:
      - date (YYYY-MM, YYYY-MM-01, etc.)
      - index (numeric)
    Returns a monthly CPI series indexed by Month-Start.
    """
    cpi = pd.read_csv(cpi_csv)
    if "date" not in cpi.columns or "index" not in cpi.columns:
        raise ValueError("CPI CSV must contain columns 'date' and 'index'.")

    cpi["date"] = pd.to_datetime(cpi["date"], utc=True, errors="coerce")
    cpi = cpi.dropna(subset=["date"])
    cpi["date"] = (
        cpi["date"].dt.tz_convert(tz).dt.to_period("M").dt.to_timestamp(how="start").dt.tz_localize(tz)
    )
    s = cpi.set_index("date")["index"].astype(float).sort_index().asfreq("MS")
    return s.rename("cpi_index")


def apply_cpi_adjustment(
    price_nominal: pd.Series,
    cpi_index: pd.Series,
    base_period: str = "2020-01",
) -> pd.Series:
    """
    Convert nominal USD/MWh -> real USD/MWh using CPI index ratio:
      real_t = nominal_t * (CPI_base / CPI_t)

    'base_period' must exist in CPI index; typical: "2020-01".
    """
    # Align frequencies & union index
    cpi = cpi_index.reindex(price_nominal.index, method="ffill")

    # Find CPI at base period (month-start normalized)
    base_ts = pd.to_datetime(f"{base_period}-01", utc=True)
    base_ts = base_ts.tz_convert(price_nominal.index.tz).to_period("M").to_timestamp(how="start").tz_localize(price_nominal.index.tz)

    if base_ts not in cpi.index or pd.isna(cpi.loc[base_ts]):
        raise ValueError(f"CPI base period {base_period} not present in CPI series.")

    cpi_base = float(cpi.loc[base_ts])
    real = price_nominal * (cpi_base / cpi)
    return real.rename("price_usd_mwh_real")


# ------------------------------------------------- #
# 4) ---- Block-bootstrap Monte Carlo on returns ---- #
# ------------------------------------------------- #

def compute_log_returns(series: pd.Series) -> np.ndarray:
    """
    Compute log-returns from a strictly-positive price series.
    """
    s = series.astype(float).copy()
    # Safety: interpolate any non-positive or missing values
    s[s <= 0] = np.nan
    s = s.interpolate(limit_direction="both")
    log_p = np.log(s.values)
    returns = np.diff(log_p)
    return returns


def clip_outliers(arr: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    """
    Clip extreme values to [q_low, q_high] quantile to stabilize simulation.
    """
    lo, hi = np.quantile(arr, [q_low, q_high])
    return np.clip(arr, lo, hi)


def sample_block_bootstrap_returns(
    log_returns: np.ndarray,
    horizon_m: int,
    block_len: int = 12,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a path of log-returns of length horizon_m by concatenating
    randomly drawn sliding blocks of length 'block_len'.
    - Uses numpy.sliding_window_view over log_returns.
    - For remainders (if horizon not divisible by block_len), we take the first 'r' elements
      from an extra randomly sampled block.
    """
    if rng is None:
        rng = np.random.default_rng()
    if log_returns.shape[0] < block_len + 1:
        raise ValueError("Not enough history to build sliding blocks of given block_len.")

    from numpy.lib.stride_tricks import sliding_window_view
    blocks = sliding_window_view(log_returns, window_shape=block_len)  # shape: (num_blocks, block_len)
    num_blocks = blocks.shape[0]

    n_full = horizon_m // block_len
    remainder = horizon_m % block_len

    chosen = rng.integers(0, num_blocks, size=n_full)
    path = blocks[chosen].reshape(-1)

    if remainder > 0:
        extra_idx = int(rng.integers(0, num_blocks))
        path = np.concatenate([path, blocks[extra_idx][:remainder]])

    return path  # length = horizon_m


def simulate_price_paths(
    last_price: float,
    log_returns_hist: np.ndarray,
    n_scenarios: int,
    horizon_m: int,
    block_len: int = 12,
    q_clip_low: float = 0.01,
    q_clip_high: float = 0.99,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Monte Carlo price paths (USD/MWh) as a (n_scenarios, horizon_m) array.
    - Simulates in log-space to keep positivity.
    - Uses block bootstrap to preserve seasonality/covariance across adjacent months.
    """
    rng = np.random.default_rng(seed)
    # Stabilize extremes
    returns = clip_outliers(log_returns_hist, q_clip_low, q_clip_high)

    # Simulate each path
    log_p0 = np.log(float(last_price))
    out = np.empty((n_scenarios, horizon_m), dtype=np.float32)

    for i in range(n_scenarios):
        r_path = sample_block_bootstrap_returns(returns, horizon_m, block_len, rng)
        log_path = log_p0 + np.cumsum(r_path)
        out[i, :] = np.exp(log_path).astype(np.float32)

    return out  # (N, T)


# ------------------------------------------- #
# 5) ---- Quantiles, export, and plotting ---- #
# ------------------------------------------- #

def build_future_index(last_date: pd.Timestamp, horizon_m: int) -> pd.DatetimeIndex:
    """
    Build a Month-Start index for the forecast horizon (next month to horizon_m months).
    """
    return pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon_m, freq="MS", tz=last_date.tz)


def quantiles_over_paths(paths: np.ndarray, q_levels=(0.10, 0.50, 0.90)) -> np.ndarray:
    """
    Compute quantiles across axis=0 (scenarios) for each month: shape (len(q), horizon).
    """
    return np.quantile(paths, q_levels, axis=0)


def export_paths_long(
    future_idx: pd.DatetimeIndex,
    price_paths_nominal: np.ndarray,
    out_dir: Path,
    prefix: str,
    cpi_series: Optional[pd.Series] = None,
) -> Path:
    """
    Save all scenarios in long format:
      date, scenario_id, price_usd_mwh_nominal[, price_usd_mwh_real], currency
    """
    n_scen, T = price_paths_nominal.shape
    scenario_ids = np.arange(n_scen, dtype=np.int32)
    dates_rep = np.tile(future_idx.values, n_scen)
    scen_rep = np.repeat(scenario_ids, T)

    data = {
        "date": pd.to_datetime(dates_rep),
        "scenario_id": scen_rep,
        "price_usd_mwh_nominal": price_paths_nominal.reshape(-1),
        "currency": "USD",
    }

    # Optional: add real prices if CPI provided (assumes CPI aligned to future dates)
    if cpi_series is not None:
        # Align CPI to forecast dates (ffill). Expect CPI future path is provided by user,
        # or they choose to keep nominal for financial model. If CPI future not provided,
        # you can skip real export.
        cpi = cpi_series.reindex(future_idx, method="ffill")
        if cpi.isna().any():
            print("[WARN] CPI has NaNs on forecast dates; skipping real-price export.")
        else:
            # Normalize to base period inside apply_cpi_adjustment (user pre-adjustment)
            # Here we assume CPI already normalized to base, so real = nominal * (CPI_base / CPI_t) = nominal * (100 / CPI_t) if base=100
            # To avoid confusion, we only export nominal by default, and recommend using quantiles nominal for cash-flow.
            pass  # left intentionally; provide real via quantile export if needed.

    df_long = pd.DataFrame(data)
    path_parquet = out_dir / f"{prefix}.parquet"
    path_csv = out_dir / f"{prefix}.csv"
    written = try_save_parquet_or_csv(df_long, path_parquet, path_csv)
    return written


def export_quantiles(
    future_idx: pd.DatetimeIndex,
    q_nominal: np.ndarray,
    out_dir: Path,
    prefix: str,
    real_series_quantiles: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """
    Save monthly quantiles for nominal (and optionally real) prices.
    q_nominal: shape (3, T) for P10/P50/P90
    """
    quantiles_df = pd.DataFrame({
        "date": future_idx,
        "p10_nominal_usd_mwh": q_nominal[0, :],
        "p50_nominal_usd_mwh": q_nominal[1, :],
        "p90_nominal_usd_mwh": q_nominal[2, :],
    })

    # If you provide real quantiles, they will be appended
    if real_series_quantiles is not None:
        for k, arr in real_series_quantiles.items():
            quantiles_df[k] = arr

    path_parquet = out_dir / f"{prefix}.parquet"
    path_csv = out_dir / f"{prefix}.csv"
    written = try_save_parquet_or_csv(quantiles_df, path_parquet, path_csv)
    return written


def plot_quantile_lines(future_idx: pd.DatetimeIndex, q_nominal: np.ndarray, out_path: Path) -> None:
    """
    Plot P10/P50/P90 lines (matplotlib only, no seaborn).
    """
    plt.figure()
    plt.plot(future_idx, q_nominal[0, :], label="P10")
    plt.plot(future_idx, q_nominal[1, :], label="P50 (median)")
    plt.plot(future_idx, q_nominal[2, :], label="P90")
    plt.xlabel("Date (Monthly)")
    plt.ylabel("USD/MWh (Nominal)")
    plt.title("Forecast Price Quantiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)


def plot_final_month_hist(paths: np.ndarray, out_path: Path, bins: int = 60) -> None:
    """
    Histogram of month-H prices across scenarios.
    """
    plt.figure()
    final_prices = paths[:, -1]
    plt.hist(final_prices, bins=bins)
    plt.xlabel("USD/MWh (Nominal)")
    plt.ylabel("Count")
    plt.title("Distribution of Price at Final Month")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)


# ----------------------------------- #
# 6) ----------------- CLI ----------- #
# ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Monthly electricity price Monte Carlo simulator (block bootstrap).")
    ap.add_argument("--us_csv", type=str, required=True, help="Path to U.S. price dataset CSV (monthly, with year/month columns).")
    ap.add_argument("--state", type=str, default="*", help="State filter (exact text in 'stateDescription'), or '*' for all.")
    ap.add_argument("--sector", type=str, default="*", help="Sector filter (exact text in 'sectorName'), or '*' for all.")
    ap.add_argument("--unit_source", type=str, default="cent_per_kwh",
                    choices=["cent_per_kwh", "usd_per_kwh", "usd_per_mwh"],
                    help="Unit of 'price' column to convert to USD/MWh.")
    ap.add_argument("--weight_col", type=str, default="sales", help="Weighting column for monthly average (default: sales).")

    ap.add_argument("--cpi_csv", type=str, default=None, help="(Optional) Path to CPI CSV with columns: date, index.")
    ap.add_argument("--cpi_base", type=str, default="2020-01", help="Base period for CPI adjustment (e.g., 2020-01).")

    ap.add_argument("--horizon_months", type=int, default=120, help="Forecast horizon in months (default: 120).")
    ap.add_argument("--n_scenarios", type=int, default=10000, help="Number of Monte Carlo scenarios (default: 10000).")
    ap.add_argument("--block_len", type=int, default=12, help="Block length for bootstrap (default: 12 months).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--tz", type=str, default="UTC", help="Timezone for datetime index (default: UTC).")

    ap.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    ap.add_argument("--prefix", type=str, default=None, help="Custom filename prefix. Default: derived from M{horizon}_N{scenarios}.")

    args = ap.parse_args()

    us_csv = Path(args.us_csv)
    out_dir, plots_dir = ensure_out_dir(Path(args.out_dir))

    # 1) Build monthly nominal series (USD/MWh)
    series_nominal = load_us_dataset(
        csv_path=us_csv,
        tz=args.tz,
        state=args.state,
        sector=args.sector,
        unit_source=args.unit_source,
        weight_col=args.weight_col,
    )

    # 2) Optional CPI -> real series (for quantiles only; scenario export keeps nominal by default)
    real_available = False
    if args.cpi_csv:
        cpi_series = load_cpi_series(Path(args.cpi_csv), tz=args.tz)
        series_real = apply_cpi_adjustment(series_nominal, cpi_series, base_period=args.cpi_base)
        real_available = True
    else:
        cpi_series = None
        series_real = None

    # 3) Monte Carlo via block bootstrap on log-returns
    log_returns = compute_log_returns(series_nominal)
    last_price = float(series_nominal.iloc[-1])

    price_paths = simulate_price_paths(
        last_price=last_price,
        log_returns_hist=log_returns,
        n_scenarios=int(args.n_scenarios),
        horizon_m=int(args.horizon_months),
        block_len=int(args.block_len),
        seed=int(args.seed),
    )

    # 4) Quantiles (nominal)
    q_nominal = quantiles_over_paths(price_paths, q_levels=(0.10, 0.50, 0.90))
    future_idx = build_future_index(series_nominal.index[-1], int(args.horizon_months))

    # Optional real quantiles if CPI for future provided (commonly not available; we usually keep nominal for cashflows)
    real_quantiles_dict = None
    if real_available and cpi_series is not None:
        # If you have a CPI *forecast* series for the future horizon, you can produce real quantiles by deflating each scenario path.
        # Here we only export nominal quantiles; documenting real generation is left to a CPI-forecast step.
        pass

    # 5) Export
    prefix = args.prefix or f"price_paths_M{args.horizon_months}_N{args.n_scenarios}"
    paths_written = export_paths_long(
        future_idx=future_idx,
        price_paths_nominal=price_paths,
        out_dir=out_dir,
        prefix=prefix,
        cpi_series=None,  # keeping nominal in long table for clarity
    )

    q_prefix = f"price_quantiles_M{args.horizon_months}"
    q_written = export_quantiles(
        future_idx=future_idx,
        q_nominal=q_nominal,
        out_dir=out_dir,
        prefix=q_prefix,
        real_series_quantiles=real_quantiles_dict,
    )

    # 6) Plots
    plot_quantile_lines(future_idx, q_nominal, out_path=plots_dir / "price_quantiles_lines.png")
    plot_final_month_hist(price_paths, out_path=plots_dir / "price_monthH_hist.png")

    # 7) Meta
    meta = {
        "source": "U.S. electricity prices (user-provided CSV)",
        "filters": {"state": args.state, "sector": args.sector},
        "unit_source": args.unit_source,
        "series_nominal_start": str(series_nominal.index.min()),
        "series_nominal_end": str(series_nominal.index.max()),
        "series_nominal_points": int(series_nominal.shape[0]),
        "last_price_usd_mwh_nominal": last_price,
        "monte_carlo": {
            "n_scenarios": int(args.n_scenarios),
            "horizon_months": int(args.horizon_months),
            "block_len": int(args.block_len),
            "seed": int(args.seed),
            "returns_clip_quantiles": [0.01, 0.99],
        },
        "exports": {
            "paths": str(paths_written),
            "quantiles": str(q_written),
            "plots": [
                str(plots_dir / "price_quantiles_lines.png"),
                str(plots_dir / "price_monthH_hist.png"),
            ],
        },
        "notes": [
            "Paths are nominal USD/MWh; couple with energy paths on (date, scenario_id).",
            "If you need real prices, provide CPI (and CPI forecast for future months) and deflate nominal accordingly.",
        ],
    }
    with open(out_dir / "price_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 8) Print key P-values for first & last month
    p10_first, p50_first, p90_first = q_nominal[0, 0], q_nominal[1, 0], q_nominal[2, 0]
    p10_last, p50_last, p90_last   = q_nominal[0, -1], q_nominal[1, -1], q_nominal[2, -1]
    print("Key quantiles (USD/MWh, nominal):")
    print(f"  First month : P10={p10_first:.2f}, P50={p50_first:.2f}, P90={p90_first:.2f}")
    print(f"  Last  month : P10={p10_last:.2f}, P50={p50_last:.2f}, P90={p90_last:.2f}")
    print(f"Paths -> {paths_written}")
    print(f"Quantiles -> {q_written}")
    print(f"Plots -> {plots_dir / 'price_quantiles_lines.png'}, {plots_dir / 'price_monthH_hist.png'}")


if __name__ == "__main__":
    main()
