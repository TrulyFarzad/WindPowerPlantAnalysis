# -*- coding: utf-8 -*-
"""
wind_resource.py — Monthly production modeling for a wind plant (MWh),
using SCADA/forecasting datasets when available, with plant-level
availability/losses/degradation and Monte Carlo uncertainty around a
seasonal baseline.

Outputs (saved next to config.yaml):
    1) wind_hist_cf.png
    2) wind_sampler_fan_chart.png
    3) wind_sampler_heatmap.png
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Dataclasses -------------------------

@dataclass
class MCVolCalib:
    target_monthly_sigma: Optional[float] = None
    hard_min: float = 0.5
    hard_max: float = 2.0

@dataclass
class StudentT:
    nu: float = 5.0

@dataclass
class WindUncertaintyCfg:
    sampler: str = "bootstrap"          # "bootstrap" | "student_t" | "normal"
    by_bucket: str = "month_of_year"    # "month_of_year" | "season3" | "none"
    scale: float = 1.0
    calibrate_sigma: Optional[MCVolCalib] = None
    student_t: Optional[StudentT] = None

@dataclass
class WindFallbackCF:
    mean: float = 0.35
    std: float = 0.05
    min: float = 0.05
    max: float = 0.60

@dataclass
class WindCfg:
    scada_path: Optional[str]
    forecasting_path: Optional[str]
    prefer_source: str = "scada"
    rated_kw: Optional[float] = None
    use_theoretical_power: bool = True
    cf_floor: float = 0.02
    cf_cap: float = 0.65
    uncertainty: WindUncertaintyCfg = field(default_factory=WindUncertaintyCfg)
    fallback_cf: WindFallbackCF = field(default_factory=WindFallbackCF)


# ------------------------- Helpers -------------------------

def _month_key(ts: pd.Timestamp) -> Tuple[int, int]:
    return ts.year, ts.month

def _hours_in_month(ts: pd.Timestamp) -> float:
    start = pd.Timestamp(ts.year, ts.month, 1)
    end = (start + pd.offsets.MonthEnd(1))
    return (end - start + pd.Timedelta(days=1)).total_seconds() / 3600.0  # inclusive month end

def _bucket_key(mo: int, mode: str) -> int:
    if mode == "month_of_year":
        return mo
    if mode == "season3":
        if mo in [12, 1, 2]:
            return 0  # cold
        if mo in [6, 7, 8]:
            return 2  # hot
        return 1      # shoulder
    return -1

def _calibrate_scale(hist_sigma: float, cfg: WindUncertaintyCfg) -> float:
    cal = cfg.calibrate_sigma
    if not cal or cal.target_monthly_sigma is None or hist_sigma <= 0:
        return float(cfg.scale)
    s = float(cal.target_monthly_sigma) / float(hist_sigma)
    s = max(float(cal.hard_min), min(float(cal.hard_max), s))
    return s

def _safe_percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return np.nan
    return float(np.nanpercentile(a, q))

# ------------------------- Loaders -------------------------

def _load_scada_excel(path: str) -> pd.DataFrame:
    """
    Reads SCADA T1.xlsx. Expected columns (robust to variations):
      - timestamp: 'Date/Time' or 'Date'/'Time'
      - 'LV ActivePower (kW)' (or similar)
      - 'Wind Speed (m/s)' (optional)
      - 'Theoretical_Power_Curve (KWh)' (optional; per 10-min energy)
    Returns a tidy frame: ['timestamp','P_kW','WS_ms','TheoP_kW'] at 10-min resolution.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "P_kW", "WS_ms", "TheoP_kW"])
    df = pd.read_excel(path)
    cols = {c.strip(): c for c in df.columns}

    # Timestamp
    ts = None
    for key in ["Date/Time", "Timestamp", "DATE_TIME", "Date Time", "Date", "Datetime"]:
        m = [c for c in cols if key.lower() in c.lower()]
        if m:
            ts = pd.to_datetime(df[cols[m[0]]], errors="coerce")
            break
    if ts is None and "Date" in cols and "Time" in cols:
        ts = pd.to_datetime(
            df[cols["Date"]].astype(str) + " " + df[cols["Time"]].astype(str),
            errors="coerce"
        )
    if ts is None:
        # fallback first column
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    # Active Power (kW)
    pcols = [c for c in cols if "active" in c.lower() and "power" in c.lower()]
    P_kW = pd.to_numeric(df[cols[pcols[0]]], errors="coerce") if pcols else np.nan

    # Wind Speed (m/s)
    wcols = [c for c in cols if "wind" in c.lower() and "speed" in c.lower()]
    WS_ms = pd.to_numeric(df[cols[wcols[0]]], errors="coerce") if wcols else np.nan

    # Theoretical curve (KWh per 10-min) -> convert to kW by ×6
    tcols = [c for c in cols if "theoretical" in c.lower() or "curve" in c.lower()]
    TheoP_kW = None
    if tcols:
        th = pd.to_numeric(df[cols[tcols[0]]], errors="coerce")
        TheoP_kW = th * 6.0  # kWh per 10-min -> kW

    out = pd.DataFrame({
        "timestamp": ts,
        "P_kW": P_kW,
        "WS_ms": WS_ms,
        "TheoP_kW": TheoP_kW if TheoP_kW is not None else np.nan
    })
    out = out.dropna(subset=["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)

def _load_forecasting_excel(path: str) -> pd.DataFrame:
    """
    Reads Turbine_Data.xlsx (10-min features). If it contains a power/energy column,
    we can derive CF; otherwise we mainly use it for wind-speed-based fallback later.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    # Try to guess timestamp
    ts_col = None
    for key in ["Date/Time", "Timestamp", "Datetime", "date", "time", "Date"]:
        m = [c for c in df.columns if key.lower() in str(c).lower()]
        if m:
            ts_col = m[0]; break
    ts = pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.to_datetime(df.iloc[:,0], errors="coerce")

    # Guess power/energy column if exists
    pcol = None
    for key in ["ActivePower", "Power", "kW", "MW", "Energy", "kWh", "MWh"]:
        m = [c for c in df.columns if key.lower() in str(c).lower()]
        if m:
            pcol = m[0]; break
    P = pd.to_numeric(df[pcol], errors="coerce") if pcol else np.nan

    out = pd.DataFrame({"timestamp": ts, "value": P})
    out = out.dropna(subset=["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)

# ------------------------- SCADA → Monthly CF -------------------------

def _infer_rated_kw(scada: pd.DataFrame, rated_kw_cfg: Optional[float], use_theo: bool) -> float:
    if rated_kw_cfg and rated_kw_cfg > 0:
        return float(rated_kw_cfg)
    # Prefer theoretical curve if present
    if use_theo and "TheoP_kW" in scada.columns and scada["TheoP_kW"].notna().any():
        return float(np.nanmax(scada["TheoP_kW"].values))
    # Else use empirical high quantile of Active Power
    return float(np.nanpercentile(scada["P_kW"].values, 99.5))

def _scada_monthly_cf(scada: pd.DataFrame, rated_kw: float) -> pd.DataFrame:
    """
    Compute monthly capacity factor from 10-min ActivePower (kW).
      CF_m = sum(P_kW_i * (10/60)) / (rated_kw * hours_in_month)
    Returns ['date','cf'] with date at month-end.
    """
    x = scada.dropna(subset=["timestamp", "P_kW"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["date","cf"])
    x["date"] = pd.to_datetime(x["timestamp"]).dt.to_period("M").dt.to_timestamp("M")
    # Energy per row in kWh over 10-min
    x["e_kWh"] = pd.to_numeric(x["P_kW"], errors="coerce") * (10.0/60.0)
    # Aggregate by month
    agg = x.groupby("date", as_index=False).agg(e_kWh=("e_kWh","sum"))
    # Hours per calendar month
    agg["hours"] = agg["date"].apply(_hours_in_month)
    agg["cf"] = np.clip((agg["e_kWh"] / (rated_kw * agg["hours"])), 0.0, 1.2)  # allow slight >1 due to noise
    agg["cf"] = agg["cf"].clip(upper=1.0)
    return agg[["date","cf"]].sort_values("date").reset_index(drop=True)

def _seasonal_baseline_from_cf(cf_monthly: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return seasonal baseline (12 values) and log-residuals."""
    if cf_monthly is None or cf_monthly.empty:
        return np.full(12, np.nan), np.array([])
    cf = cf_monthly.copy()
    cf["m"] = cf["date"].dt.month
    seas = cf.groupby("m")["cf"].mean().reindex(range(1,13)).interpolate().to_numpy()
    # Avoid zeros in log
    eps = 1e-6
    cf = cf.merge(pd.DataFrame({"m":np.arange(1,13), "seas":seas}), on="m", how="left")
    cf["resid"] = np.log((cf["cf"]+eps) / (cf["seas"]+eps))
    return seas, cf["resid"].to_numpy()

# ------------------------- Plant baseline & Monte Carlo -------------------------

def _degradation_factor(t_month: int, deg_per_year: float) -> float:
    years = t_month // 12
    return (1.0 - float(deg_per_year)) ** years

def _plant_mwh_from_cf(cf: float, capacity_mw: float, hours_in_month: float) -> float:
    return float(capacity_mw) * 1000.0 * hours_in_month * float(cf) / 1000.0  # kWh->MWh

def _draw_residuals(n: int, pool: np.ndarray, rng: np.random.Generator, cfg: WindUncertaintyCfg) -> np.ndarray:
    if cfg.sampler == "student_t":
        nu = float((cfg.student_t.nu if cfg.student_t else 5.0))
        z = rng.standard_t(df=nu, size=n)
        # scale later by histogram σ
        return z
    if cfg.sampler == "normal":
        return rng.normal(size=n)
    # bootstrap
    if pool.size == 0:
        return rng.normal(size=n)
    return rng.choice(pool, size=n, replace=True)

def _bucket_residuals(res: np.ndarray, months: np.ndarray, mode: str) -> Dict[int, np.ndarray]:
    if res.size == 0 or mode == "none":
        return {-1: res}
    buckets: Dict[int, List[float]] = {}
    for r, mo in zip(res, months):
        k = _bucket_key(int(mo), mode)
        buckets.setdefault(k, []).append(r)
    return {k: np.array(v, dtype=float) for k, v in buckets.items()}

def sample_production_paths_monthly(
    n_iter: int,
    months_h: int,
    rng: np.random.Generator,
    cfg_yaml: Dict
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      - paths_mwh:  (n_iter, months_h) plant monthly energy (MWh)
      - baseline_mwh: (months_h,) deterministic baseline (seasonal CF × plant factors)
      - hist_cf_monthly: DataFrame with ['date','cf'] if SCADA available
    """
    plant = cfg_yaml.get("plant", {})
    capacity_mw = float(plant.get("capacity_mw", 20.0))
    availability = float(plant.get("availability", 0.97))
    losses = float(plant.get("losses_fraction", 0.10))
    deg = float(plant.get("degradation_per_year", 0.007))

    wind = cfg_yaml.get("wind", {})
    wcfg = WindCfg(
        scada_path=wind.get("scada_path"),
        forecasting_path=wind.get("forecasting_path"),
        prefer_source=str(wind.get("prefer_source","scada")),
        rated_kw=(None if wind.get("rated_kw") in [None, "null"] else float(wind.get("rated_kw"))),
        use_theoretical_power=bool(wind.get("use_theoretical_power", True)),
        cf_floor=float(wind.get("cf_floor", 0.02)),
        cf_cap=float(wind.get("cf_cap", 0.65)),
        uncertainty=WindUncertaintyCfg(
            sampler=wind.get("uncertainty",{}).get("sampler","bootstrap"),
            by_bucket=wind.get("uncertainty",{}).get("by_bucket","month_of_year"),
            scale=float(wind.get("uncertainty",{}).get("scale",1.0)),
            calibrate_sigma=(MCVolCalib(
                target_monthly_sigma=(None if wind.get("uncertainty",{}).get("calibrate_sigma",{}).get("target_monthly_sigma") in [None,"null"] else float(wind.get("uncertainty",{}).get("calibrate_sigma",{}).get("target_monthly_sigma"))),
                hard_min=float(wind.get("uncertainty",{}).get("calibrate_sigma",{}).get("hard_min",0.5)),
                hard_max=float(wind.get("uncertainty",{}).get("calibrate_sigma",{}).get("hard_max",2.0)),
            ) if "calibrate_sigma" in wind.get("uncertainty",{}) else None),
            student_t=(StudentT(nu=float(wind.get("uncertainty",{}).get("student_t",{}).get("nu",5.0))) if wind.get("uncertainty",{}).get("sampler")=="student_t" else None)
        ),
        fallback_cf=WindFallbackCF(
            mean=float(wind.get("fallback_cf",{}).get("mean",0.35)),
            std=float(wind.get("fallback_cf",{}).get("std",0.05)),
            min=float(wind.get("fallback_cf",{}).get("min",0.05)),
            max=float(wind.get("fallback_cf",{}).get("max",0.60)),
        )
    )

    # ---------- 1) Historical monthly CF from SCADA if available ----------
    hist_cf_monthly = pd.DataFrame(columns=["date","cf"])
    rated_kw = None
    if wcfg.scada_path and os.path.exists(wcfg.scada_path):
        scada = _load_scada_excel(wcfg.scada_path)
        if not scada.empty and scada["P_kW"].notna().any():
            rated_kw = _infer_rated_kw(scada, wcfg.rated_kw, wcfg.use_theoretical_power)
            hist_cf_monthly = _scada_monthly_cf(scada, rated_kw)

    # ---------- 2) Seasonal baseline CF ----------
    if not hist_cf_monthly.empty:
        seasonal_cf, resid_log = _seasonal_baseline_from_cf(hist_cf_monthly)
        months_hist = hist_cf_monthly["date"].dt.month.to_numpy()
        sigma_hist = float(np.std(resid_log, ddof=1)) if resid_log.size > 1 else 0.0
    else:
        # No history: flat seasonal profile at fallback mean
        seasonal_cf = np.full(12, float(wcfg.fallback_cf.mean))
        resid_log = np.array([])
        months_hist = np.array([], dtype=int)
        sigma_hist = 0.0

    # Plant-level multiplicative factor (availability & losses)
    plant_factor = float(availability) * (1.0 - float(losses))

    # Baseline CF path for horizon (seasonal × plant factor × degradation-by-year)
    baseline_cf = np.zeros(months_h, dtype=float)
    baseline_mwh = np.zeros(months_h, dtype=float)
    for t in range(months_h):
        mo = (t % 12) + 1
        cf0 = seasonal_cf[mo-1]
        cf0 = np.clip(cf0, wcfg.cf_floor, wcfg.cf_cap)
        cf0 *= plant_factor
        cf0 *= _degradation_factor(t, deg)
        baseline_cf[t] = cf0
        # hours in the t-th month — approximate with 30.4375*24 if Jalali not mapped here
        # We'll compute hours from Gregorian series in __main__ when plotting; here use 30*24 for calc
        baseline_mwh[t] = cf0 * capacity_mw * 30.4375 * 24.0  # ≈ average month hours

    # ---------- 3) Monte Carlo around baseline ----------
    # Residual buckets for bootstrap / controls for student_t / normal
    buckets = _bucket_residuals(resid_log, months_hist, wcfg.uncertainty.by_bucket)
    scale = _calibrate_scale(sigma_hist, wcfg.uncertainty)
    if scale == 0.0:
        paths_mwh = np.tile(baseline_mwh, (n_iter, 1))
        return paths_mwh, baseline_mwh, hist_cf_monthly

    # Draw log-residuals by month bucket and exponentiate around baseline CF
    paths_mwh = np.zeros((n_iter, months_h), dtype=float)
    for t in range(months_h):
        mo = (t % 12) + 1
        key = _bucket_key(mo, wcfg.uncertainty.by_bucket)
        pool = buckets.get(key, resid_log)
        z = _draw_residuals(n_iter, pool, rng, wcfg.uncertainty)
        # If student_t/normal, scale to hist sigma; if bootstrap, scale acts as multiplier of hist σ implicitly
        if wcfg.uncertainty.sampler in ["student_t", "normal"]:
            z = z * (sigma_hist if sigma_hist > 0 else 1.0)
        eps = z * float(scale)
        cf_t = baseline_cf[t] * np.exp(eps)
        cf_t = np.clip(cf_t, wcfg.cf_floor, wcfg.cf_cap)
        paths_mwh[:, t] = cf_t * capacity_mw * 30.4375 * 24.0

    return paths_mwh, baseline_mwh, hist_cf_monthly

# ------------------------- __main__: simulate & plot -------------------------

def _load_yaml_config(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Optional Jalali month helper (re-using approach from price_model)
try:
    import jdatetime
    HAS_JDATE = True
except Exception:
    HAS_JDATE = False

def _jalali_month_range(start_y: int, start_m: int, end_y: int, end_m: int) -> List[pd.Timestamp]:
    if not HAS_JDATE:
        months = (end_y - start_y) * 12 + (end_m - start_m) + 1
        start = pd.Timestamp.today().normalize().replace(day=1)
        return list(pd.date_range(start=start, periods=months, freq="M"))
    out: List[pd.Timestamp] = []
    y, m = start_y, start_m
    while (y < end_y) or (y == end_y and m <= end_m):
        g = jdatetime.date(y, m, 1).togregorian()
        ts = pd.Timestamp(g.year, g.month, 1) + pd.offsets.MonthEnd(0)
        out.append(ts)
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out

def _percentiles_matrix(paths: np.ndarray, qs: List[float]) -> np.ndarray:
    return np.percentile(paths, qs, axis=0)

if __name__ == "__main__":
    CODEBASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(CODEBASE, "config.yaml")
    cfg_yaml = _load_yaml_config(CONFIG_PATH)

    # Horizon: 1404/01 → 1414/12 (132 months)
    start_jy, start_jm = 1404, 1
    end_jy, end_jm = 1414, 12
    horizon_months = (end_jy - start_jy) * 12 + (end_jm - start_jm) + 1
    x_dates = _jalali_month_range(start_jy, start_jm, end_jy, end_jm)

    n_iter = int(cfg_yaml.get("monte_carlo", {}).get("iterations", 1000))
    seed = int(cfg_yaml.get("monte_carlo", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed)

    # Simulate
    paths_mwh, baseline_mwh, hist_cf = sample_production_paths_monthly(
        n_iter=n_iter, months_h=horizon_months, rng=rng, cfg_yaml=cfg_yaml
    )

    # ---- Plot 0: Historical CF (from SCADA) ----
    if hist_cf is not None and not hist_cf.empty:
        plt.figure(figsize=(12, 3.5))
        plt.plot(hist_cf["date"], hist_cf["cf"], label="Historical CF (monthly)")
        plt.title("Historical Monthly Capacity Factor (SCADA)")
        plt.xlabel("Month (Gregorian)")
        plt.ylabel("Capacity Factor (ratio)")
        plt.ylim(0, 1.0)
        plt.legend()
        f0 = os.path.join(CODEBASE, "wind_hist_cf.png")
        plt.tight_layout()
        plt.savefig(f0, dpi=150)

    # ---- Plot 1: Fan chart of plant monthly MWh ----
    qs = [5, 50, 95]
    qmat = _percentiles_matrix(paths_mwh, qs)
    plt.figure(figsize=(12, 4))
    plt.plot(x_dates, baseline_mwh, label="Baseline (seasonal × availability × losses × degradation)")
    plt.plot(x_dates, qmat[1], label="P50")
    plt.fill_between(x_dates, qmat[0], qmat[2], alpha=0.2, label="P5–P95")
    plt.title("Monthly Energy Production (MWh) — Baseline + Stochastic Fan")
    plt.xlabel("Month (Gregorian)")
    plt.ylabel("Energy (MWh)")
    plt.legend()
    f1 = os.path.join(CODEBASE, "wind_sampler_fan_chart.png")
    plt.tight_layout()
    plt.savefig(f1, dpi=150)

    # ---- Plot 2: Percentile heatmap ----
    qs_dense = list(range(5, 100, 5))
    qmat_dense = _percentiles_matrix(paths_mwh, qs_dense)
    plt.figure(figsize=(12, 5))
    plt.imshow(qmat_dense, aspect="auto", origin="lower")
    plt.title("Percentile Heatmap of Simulated Monthly Energy (MWh)")
    plt.xlabel("Time (months 1404→1414)")
    plt.ylabel("Percentile (5→95)")
    xticks = np.linspace(0, horizon_months - 1, 12, dtype=int)
    plt.xticks(xticks, [x_dates[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
    yticks = np.arange(len(qs_dense))
    plt.yticks(yticks, [str(q) for q in qs_dense])
    f2 = os.path.join(CODEBASE, "wind_sampler_heatmap.png")
    plt.tight_layout()
    plt.savefig(f2, dpi=150)

    print("Saved:", (f0 if 'f0' in locals() else "(no historical CF plot)"))
    print("Saved:", f1)
    print("Saved:", f2)
