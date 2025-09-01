# -*- coding: utf-8 -*-
"""
wind_resource.py — Monthly production modeling (MWh) from:
  - SCADA (T1.csv) -> LOWESS empirical power curve + historical monthly CF
  - Turbine_Data.csv -> extra wind speed for Weibull fit
  - Weibull per month-of-year over wind speed; sampling CF per month
  - Availability, losses, degradation applied

Outputs next to config.yaml:
  - wind_power_curve.png
  - wind_hist_cf.png
  - wind_sampler_fan_chart.png
  - wind_sampler_heatmap.png
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------- Dataclasses -------------------------

@dataclass
class ScadaFilters:
    ws_min: float = 0.0
    ws_max: float = 35.0
    p_min_kw: float = 0.0
    p_max_kw: Optional[float] = None  # if None -> infer from data (99.9%)

@dataclass
class WeibullCfg:
    fit_loc_zero: bool = True
    min_points_per_month: int = 200

@dataclass
class SimMonthCF:
    reps: int = 5000
    samples_per_month: int = 720  # ~hourly

@dataclass
class WindFallbackCF:
    mean: float = 0.35
    std: float = 0.05
    min: float = 0.10
    max: float = 0.55

@dataclass
class WindCfg:
    scada_path: Optional[str]
    forecasting_path: Optional[str]
    prefer_source: str = "auto"   # auto | scada | forecasting
    rated_kw: Optional[float] = None
    use_theoretical_power: bool = True
    power_curve_source: str = "auto"  # NEW: lowess | theoretical | auto
    scada_filters: ScadaFilters = field(default_factory=ScadaFilters)
    weibull: WeibullCfg = field(default_factory=WeibullCfg)
    simulate_month_cf: SimMonthCF = field(default_factory=SimMonthCF)
    cf_floor: float = 0.02
    cf_cap: float = 0.65
    fallback_cf: WindFallbackCF = field(default_factory=WindFallbackCF)

# ------------------------- YAML -------------------------

def _load_yaml_config(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ------------------------- Helpers -------------------------

def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1) + pd.offsets.MonthEnd(0)

def _hours_in_month(ts: pd.Timestamp) -> float:
    start = pd.Timestamp(ts.year, ts.month, 1)
    end = start + pd.offsets.MonthEnd(1)
    return (end - start).total_seconds() / 3600.0

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace(r"[^\d\.\-eE+]", "", regex=True),
        errors="coerce"
    )

# ------------------------- Robust loaders -------------------------

def _load_scada(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp","P_kW","WS_ms","TheoP_kW"])

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        try:
            xls = pd.ExcelFile(path)
            best, best_score = None, -1
            for sh in xls.sheet_names:
                tdf = pd.read_excel(xls, sheet_name=sh)
                names = [str(c).strip().lower() for c in tdf.columns]
                score = sum("power" in n for n in names) + sum("wind" in n and "speed" in n for n in names)
                if score > best_score:
                    best, best_score = tdf, score
            df = best if best is not None else pd.read_excel(path)
        except Exception:
            df = pd.read_excel(path)

    cols = {str(c).strip(): c for c in df.columns}

    # timestamp
    ts_col = None
    for key in ["Date/Time","Timestamp","Datetime","Date Time","Date"]:
        for c in cols:
            if key.lower() in c.lower():
                ts_col = cols[c]; break
        if ts_col is not None: break
    ts = pd.to_datetime(df[ts_col], errors="coerce") if ts_col is not None else pd.to_datetime(df.iloc[:,0], errors="coerce")

    # Active power (kW)
    p_col = None
    for key in ["LV ActivePower (kW)","ActivePower","Active Power","(kW)"]:
        for c in cols:
            if key.lower() in c.lower():
                p_col = cols[c]; break
        if p_col is not None: break
    P_kW = _coerce_numeric(df[p_col]) if p_col is not None else pd.Series(np.nan, index=df.index)

    # Wind speed (m/s)
    ws_col = None
    for key in ["Wind Speed (m/s)","WindSpeed","Wind Speed","WS","m/s"]:
        for c in cols:
            if key.lower() in c.lower():
                ws_col = cols[c]; break
        if ws_col is not None: break
    WS_ms = _coerce_numeric(df[ws_col]) if ws_col is not None else pd.Series(np.nan, index=df.index)

    # Theoretical power
    theo_col = None
    for key in ["Theoretical_Power_Curve", "Theoretical", "Curve"]:
        for c in cols:
            if key.lower() in c.lower():
                theo_col = cols[c]; break
        if theo_col is not None: break
    TheoP_kW = None
    if theo_col is not None:
        th = _coerce_numeric(df[theo_col])
        TheoP_kW = th * 6.0  # 10-min logging

    out = pd.DataFrame({
        "timestamp": ts,
        "P_kW": P_kW,
        "WS_ms": WS_ms,
        "TheoP_kW": TheoP_kW if TheoP_kW is not None else np.nan
    })
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def _load_forecasting_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp","WS_ms"])
    df = pd.read_csv(path)
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["timestamp","WS_ms"])

    ws_col = None
    for key in ["WindSpeed","Wind Speed","WS"]:
        m = [c for c in df.columns if key.lower() in str(c).lower()]
        if m:
            ws_col = m[0]; break
    WS_ms = _coerce_numeric(df[ws_col]) if ws_col else pd.Series(np.nan, index=df.index)

    ts0 = pd.Timestamp("2018-01-01 00:00:00")
    ts = pd.date_range(start=ts0, periods=n, freq="10min")
    out = pd.DataFrame({"timestamp": ts, "WS_ms": WS_ms})
    return out.dropna(subset=["WS_ms"]).reset_index(drop=True)

# ------------------------- Cleaning + Rated Power -------------------------

def _clean_scada(df: pd.DataFrame, flt: ScadaFilters) -> pd.DataFrame:
    x = df.copy()
    for c in ["P_kW","WS_ms","TheoP_kW"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["timestamp"])

    p_max = flt.p_max_kw
    if p_max is None and "P_kW" in x and x["P_kW"].notna().sum() > 100:
        p_max = float(np.nanpercentile(x["P_kW"], 99.9))
    if p_max is not None:
        x = x[x["P_kW"].isna() | (x["P_kW"] <= p_max)]
    if "P_kW" in x:
        x = x[x["P_kW"].isna() | (x["P_kW"] >= flt.p_min_kw)]
    if "WS_ms" in x:
        x = x[x["WS_ms"].isna() | ((x["WS_ms"] >= flt.ws_min) & (x["WS_ms"] <= flt.ws_max))]
    return x.reset_index(drop=True)

def _infer_rated_kw(scada: pd.DataFrame, cfg: WindCfg) -> float:
    if cfg.rated_kw and cfg.rated_kw > 0:
        return float(cfg.rated_kw)
    if cfg.use_theoretical_power and "TheoP_kW" in scada.columns and scada["TheoP_kW"].notna().any():
        return float(np.nanmax(scada["TheoP_kW"].values))
    if "P_kW" in scada.columns and scada["P_kW"].notna().any():
        return float(np.nanpercentile(scada["P_kW"].values, 99.5))
    return 2000.0

# ------------------------- Power curve -------------------------

def _iec_like_power_curve(rated_kw: float, v: np.ndarray) -> np.ndarray:
    v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
    p = np.zeros_like(v, dtype=float)
    mask1 = (v >= v_cut_in) & (v < v_rated)
    p[mask1] = rated_kw * ((v[mask1] - v_cut_in) / (v_rated - v_cut_in))**3
    mask2 = (v >= v_rated) & (v < v_cut_out)
    p[mask2] = rated_kw
    return p

def _fit_power_curve(scada: pd.DataFrame, rated_kw: float, source: str="auto") -> Tuple[np.ndarray, np.ndarray]:
    v_grid = np.linspace(0.0, 30.0, 400)

    if source == "theoretical" and "TheoP_kW" in scada and scada["TheoP_kW"].notna().any():
        x = scada.dropna(subset=["WS_ms", "TheoP_kW"])
        bins = np.linspace(0, 30, 60)
        labels = np.digitize(x["WS_ms"], bins) - 1
        gb = x.groupby(labels)["TheoP_kW"].mean()
        gb = gb.reindex(range(len(bins)-1), fill_value=0.0)
        p_grid = np.interp(v_grid, bins[:-1], gb.values, left=0, right=rated_kw)
        return v_grid, np.clip(p_grid, 0, rated_kw)

    if source in ["lowess", "auto"]:
        x = scada.dropna(subset=["WS_ms","P_kW"])
        if not x.empty and x["P_kW"].abs().sum() > 1e-6:
            fit = lowess(
                endog=np.clip(x["P_kW"].values, 0, rated_kw),
                exog=x["WS_ms"].values, frac=0.2, it=3, return_sorted=True
            )
            v, p = fit[:,0], fit[:,1]
            p_grid = np.interp(v_grid, v, p, left=p[0], right=p[-1])
            return v_grid, np.maximum.accumulate(np.clip(p_grid, 0, rated_kw))

    return v_grid, _iec_like_power_curve(rated_kw, v_grid)

def _p_of_v(v: np.ndarray, v_grid: np.ndarray, p_grid: np.ndarray, rated_kw: float) -> np.ndarray:
    p = np.interp(v, v_grid, p_grid, left=0.0, right=p_grid[-1])
    return np.clip(p, 0.0, rated_kw)

# ------------------------- Historical CF from SCADA -------------------------

def _scada_monthly_cf(scada: pd.DataFrame, rated_kw: float) -> pd.DataFrame:
    x = scada.dropna(subset=["timestamp","P_kW"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["date","cf"])
    x["date"] = x["timestamp"].dt.to_period("M").dt.to_timestamp("M")
    x["e_kWh"] = np.clip(x["P_kW"], 0, rated_kw) * (10.0/60.0)
    agg = x.groupby("date", as_index=False).agg(e_kWh=("e_kWh","sum"))
    agg["hours"] = agg["date"].apply(_hours_in_month)
    agg["cf"] = np.clip(agg["e_kWh"] / (rated_kw * agg["hours"]), 0.0, 1.0)
    return agg[["date","cf"]].sort_values("date").reset_index(drop=True)

# ------------------------- Weibull per month-of-year -------------------------

def _fit_weibull_per_month(ws: pd.Series, cfg: WeibullCfg) -> Dict[int, Tuple[float,float]]:
    s = pd.DataFrame({"timestamp": ws.index, "WS_ms": ws.values}).dropna()
    if s.empty:
        return {}
    s["m"] = s["timestamp"].dt.month
    out: Dict[int, Tuple[float,float]] = {}
    for m in range(1, 13):
        arr = s.loc[s["m"]==m, "WS_ms"].dropna().values
        arr = arr[(arr>=0) & (arr<=50)]
        if arr.size < cfg.min_points_per_month:
            continue
        if cfg.fit_loc_zero:
            k, loc, c = stats.weibull_min.fit(arr, floc=0)
        else:
            k, loc, c = stats.weibull_min.fit(arr)
        out[m] = (k, c)
    return out

# ------------------------- CF distributions per month -------------------------

def _simulate_month_cf_distributions(
    v_grid: np.ndarray, p_grid: np.ndarray, rated_kw: float,
    weibull_params: Dict[int, Tuple[float,float]],
    sim_cfg: SimMonthCF,
    rng: np.random.Generator
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for m in range(1, 13):
        if m not in weibull_params:
            continue
        k, c = weibull_params[m]
        vv = stats.weibull_min.rvs(k, loc=0, scale=c,
                                   size=(sim_cfg.reps, sim_cfg.samples_per_month),
                                   random_state=rng.integers(0, 2**32-1))
        pp = _p_of_v(vv, v_grid, p_grid, rated_kw)
        mean_p = pp.mean(axis=1)
        cf = np.clip(mean_p / rated_kw, 0.0, 1.0)
        out[m] = cf
    return out

# ------------------------- Main sampler -------------------------

def sample_production_paths_monthly(
    n_iter: int,
    months_h: int,
    rng: np.random.Generator,
    cfg_yaml: Dict
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[int, np.ndarray]]:
    plant = cfg_yaml.get("plant", {})
    capacity_mw = float(plant.get("capacity_mw", 20.0))
    availability = float(plant.get("availability", 0.97))
    losses = float(plant.get("losses_fraction", 0.10))
    deg = float(plant.get("degradation_per_year", 0.007))
    plant_factor = availability * (1.0 - losses)

    w = cfg_yaml.get("wind", {})
    wcfg = WindCfg(
        scada_path=w.get("scada_path"),
        forecasting_path=w.get("forecasting_path"),
        prefer_source=str(w.get("prefer_source","auto")),
        rated_kw=(None if w.get("rated_kw") in [None, "null"] else float(w.get("rated_kw"))),
        use_theoretical_power=bool(w.get("use_theoretical_power", True)),
        power_curve_source=str(w.get("power_curve_source", "auto")),
        scada_filters=ScadaFilters(
            ws_min=float(w.get("scada_filters",{}).get("ws_min",0.0)),
            ws_max=float(w.get("scada_filters",{}).get("ws_max",35.0)),
            p_min_kw=float(w.get("scada_filters",{}).get("p_min_kw",0.0)),
            p_max_kw=(None if w.get("scada_filters",{}).get("p_max_kw") in [None, "null"] else float(w.get("scada_filters",{}).get("p_max_kw")))
        ),
        weibull=WeibullCfg(
            fit_loc_zero=bool(w.get("weibull",{}).get("fit_loc_zero", True)),
            min_points_per_month=int(w.get("weibull",{}).get("min_points_per_month", 200))
        ),
        simulate_month_cf=SimMonthCF(
            reps=int(w.get("simulate_month_cf",{}).get("reps", 5000)),
            samples_per_month=int(w.get("simulate_month_cf",{}).get("samples_per_month", 720))
        ),
        cf_floor=float(w.get("cf_floor", 0.02)),
        cf_cap=float(w.get("cf_cap", 0.65)),
        fallback_cf=WindFallbackCF(
            mean=float(w.get("fallback_cf",{}).get("mean",0.35)),
            std=float(w.get("fallback_cf",{}).get("std",0.05)),
            min=float(w.get("fallback_cf",{}).get("min",0.10)),
            max=float(w.get("fallback_cf",{}).get("max",0.55)),
        )
    )

    # Load & clean SCADA
    scada = _load_scada(wcfg.scada_path)
    scada = _clean_scada(scada, wcfg.scada_filters)

    # Forecast dataset
    fore = _load_forecasting_csv(wcfg.forecasting_path)

    # Rated power
    rated_kw = _infer_rated_kw(scada, wcfg)
    print(f"[INFO] SCADA rows={len(scada)}, non-null P_kW={scada['P_kW'].notna().sum()}, non-null WS_ms={scada['WS_ms'].notna().sum()}")
    print(f"[INFO] Forecast rows={len(fore)}, non-null WS_ms={fore['WS_ms'].notna().sum()}")
    print(f"[INFO] Inferred rated_kW={rated_kw:.1f}")

    # Historical CF from SCADA
    hist_cf = _scada_monthly_cf(scada, rated_kw)

    # Power curve
    v_grid, p_grid = _fit_power_curve(scada, rated_kw, wcfg.power_curve_source)

    # Combined wind-speed series
    ws_scada = scada.dropna(subset=["WS_ms"]).set_index("timestamp")["WS_ms"] if not scada.empty else pd.Series(dtype=float)
    ws_fore  = fore.set_index("timestamp")["WS_ms"] if not fore.empty else pd.Series(dtype=float)
    if wcfg.prefer_source == "scada":
        ws_all = ws_scada
    elif wcfg.prefer_source == "forecasting":
        ws_all = ws_fore
    else:
        ws_all = pd.concat([ws_scada, ws_fore]).sort_index()

    # Weibull params
    weibull_params = _fit_weibull_per_month(ws_all, wcfg.weibull)
    if not weibull_params:
        baseline_mwh = np.zeros(months_h)
        paths_mwh = np.zeros((n_iter, months_h))
        mu, sd = wcfg.fallback_cf.mean, wcfg.fallback_cf.std
        for t in range(months_h):
            years = t // 12
            cf = np.clip(np.random.normal(mu, sd), wcfg.fallback_cf.min, wcfg.fallback_cf.max)
            cf *= plant_factor * ((1.0 - deg) ** years)
            mwh = cf * capacity_mw * 30.4375 * 24.0
            baseline_mwh[t] = mwh
            paths_mwh[:, t] = mwh
        return paths_mwh, baseline_mwh, hist_cf, {}

    cf_dist_by_month = _simulate_month_cf_distributions(v_grid, p_grid, rated_kw, weibull_params, wcfg.simulate_month_cf, np.random.default_rng(1234))

    baseline_cf_moy = np.array([np.nanmean(cf_dist_by_month[m]) if m in cf_dist_by_month else np.nan for m in range(1,13)])
    s = pd.Series(baseline_cf_moy).interpolate(limit_direction="both")
    baseline_cf_moy = np.clip(s.to_numpy(), wcfg.cf_floor, wcfg.cf_cap)

    baseline_mwh = np.zeros(months_h)
    paths_mwh = np.zeros((n_iter, months_h))
    rng_local = np.random.default_rng(int(cfg_yaml.get("monte_carlo",{}).get("random_seed",42)))

    for t in range(months_h):
        mo = (t % 12) + 1
        years = t // 12
        cf_draws = cf_dist_by_month.get(mo, np.array([wcfg.fallback_cf.mean]))
        cf_s = rng_local.choice(cf_draws, size=n_iter, replace=True)
        cf_s = np.clip(cf_s, wcfg.cf_floor, wcfg.cf_cap)
        cf_s *= plant_factor * ((1.0 - deg) ** years)
        mwh = cf_s * capacity_mw * 30.4375 * 24.0
        paths_mwh[:, t] = mwh

        base_cf = baseline_cf_moy[mo-1] * plant_factor * ((1.0 - deg) ** years)
        baseline_mwh[t] = base_cf * capacity_mw * 30.4375 * 24.0

    return paths_mwh, baseline_mwh, hist_cf, cf_dist_by_month

# ------------------------- Plot helpers -------------------------

def _percentiles_matrix(paths: np.ndarray, qs: List[float]) -> np.ndarray:
    return np.percentile(paths, qs, axis=0)

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
        if m == 12: y += 1; m = 1
        else: m += 1
    return out

# ------------------------- __main__ -------------------------

if __name__ == "__main__":
    CODEBASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(CODEBASE, "config.yaml")
    cfg_yaml = _load_yaml_config(CONFIG_PATH)

    start_jy, start_jm = 1404, 1
    end_jy, end_jm = 1414, 12
    horizon_months = (end_jy - start_jy) * 12 + (end_jm - start_jm) + 1
    x_dates = _jalali_month_range(start_jy, start_jm, end_jy, end_jm)

    n_iter = int(cfg_yaml.get("monte_carlo", {}).get("iterations", 1000))
    seed = int(cfg_yaml.get("monte_carlo", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed)

    paths_mwh, baseline_mwh, hist_cf, _ = sample_production_paths_monthly(
        n_iter=n_iter, months_h=horizon_months, rng=rng, cfg_yaml=cfg_yaml
    )

    if hist_cf is not None and not hist_cf.empty:
        plt.figure(figsize=(12, 3.2))
        plt.plot(hist_cf["date"], hist_cf["cf"], label="Historical CF (monthly)")
        plt.title("Historical Monthly Capacity Factor (SCADA)")
        plt.xlabel("Month (Gregorian)"); plt.ylabel("Capacity Factor"); plt.ylim(0, 1.0)
        plt.legend(); f0 = os.path.join(CODEBASE, "wind_hist_cf.png")
        plt.tight_layout(); plt.savefig(f0, dpi=150)

    scada_plot = _clean_scada(_load_scada(cfg_yaml.get("wind",{}).get("scada_path","")), ScadaFilters())
    rated_plot = _infer_rated_kw(scada_plot, WindCfg(None,None))
    v_grid, p_grid = _fit_power_curve(scada_plot, rated_plot)
    plt.figure(figsize=(8,4))
    plt.plot(v_grid, p_grid, label="Power curve (LOWESS or IEC fallback)")
    plt.title("Empirical Power Curve (kW vs m/s)")
    plt.xlabel("Wind speed (m/s)"); plt.ylabel("Power (kW)"); plt.legend()
    fpc = os.path.join(CODEBASE, "wind_power_curve.png")
    plt.tight_layout(); plt.savefig(fpc, dpi=150)

    qs = [5, 50, 95]
    qmat = _percentiles_matrix(paths_mwh, qs)
    plt.figure(figsize=(12, 4))
    plt.plot(x_dates, baseline_mwh, label="Baseline (seasonal × availability × losses × degradation)", zorder=3)
    plt.plot(x_dates, qmat[1], label="P50")
    plt.fill_between(x_dates, qmat[0], qmat[2], alpha=0.2, label="P5–P95")
    plt.title("Monthly Energy Production (MWh) — Baseline + Stochastic Fan")
    plt.xlabel("Month (Gregorian)"); plt.ylabel("Energy (MWh)"); plt.legend()
    f1 = os.path.join(CODEBASE, "wind_sampler_fan_chart.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150)

    qs_dense = list(range(5, 100, 5))
    qmat_dense = _percentiles_matrix(paths_mwh, qs_dense)
    plt.figure(figsize=(12, 5))
    plt.imshow(qmat_dense, aspect="auto", origin="lower")
    plt.title("Percentile Heatmap of Simulated Monthly Energy (MWh)")
    plt.xlabel("Time (months 1404→1414)"); plt.ylabel("Percentile (5→95)")
    xticks = np.linspace(0, horizon_months - 1, 12, dtype=int)
    plt.xticks(xticks, [x_dates[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
    yticks = np.arange(len(qs_dense)); plt.yticks(yticks, [str(q) for q in qs_dense])
    f2 = os.path.join(CODEBASE, "wind_sampler_heatmap.png")
    plt.tight_layout(); plt.savefig(f2, dpi=150)

    print("Saved:", (f0 if 'f0' in locals() else "(no historical CF plot)"))
    print("Saved:", fpc)
    print("Saved:", f1)
    print("Saved:", f2)
