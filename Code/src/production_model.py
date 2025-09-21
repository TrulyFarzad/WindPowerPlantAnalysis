# modeling/production_model.py
# -*- coding: utf-8 -*-
"""
Mode A (Weibull-only) + Mode B (SCADA-calibrated) + Mode C (Hybrid Forecasting)
Drop-in replacement for wind_resource.sample_production_paths_monthly

- Inputs (from config.yaml):
  wind:
    mode: "weibull" | "scada_calibrated" | "hybrid_forecast"
    weibull_csv_path: "C:\...\weibull_monthly.csv"  # columns: city, month, k, c [, z_ref]
    city: "Khaf"
    hub_height_m: 100
    alpha: 0.14
    turbine_name: "V112/3000"                         # turbine_type from windpowerlib
    power_curve_source: "auto"                        # auto | windpowerlib | iec
    turbine_library_csv: null                          # optional custom CSV library
    turbulence_intensity: 0.0..0.2
    diurnal_enable: true/false
    ar1_phi: 0.85
    samples_per_month: 720
  scada:
    calibration_profile_path: "C:\...\calibration_profile.json"  # Mode B
  forecasting:                                      # Mode C (optional)
    enabled: true
    file: "C:\...\Turbine_Data.csv"
    timestamp_col: "Unnamed: 0"                        # fallback aware
    value_pref: ["WindSpeed", "ActivePower", "wind_speed", "power"]
    by_month: true
    freq: "H"
    min_days_per_month: 5
    smooth_hours: 3
  plant:
    capacity_mw: 20
    availability: 0.97
    losses_fraction: 0.10
    degradation_per_year: 0.007
  project:
    start_year: 2010
    start_month: 1
    years: 10

- API:
  sample_production_paths_monthly(n_iter, months_h, rng, cfg_yaml)
    -> (energy_paths_mwh: (N,T), cf_quantiles: (3,T), meta: dict, charts: dict)

No economics here — just MWh paths compatible with existing monte_carlo.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import os
import json
import calendar
import numpy as np
import pandas as pd

# ------------------------- utils -------------------------

def _hours_in_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1] * 24


def _month_index(start_year: int, start_month: int, months: int) -> List[Tuple[int, int]]:
    y, m = start_year, start_month
    out = []
    for _ in range(months):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def _std_month(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip().lower()
    names = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11,
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
        "june": 6, "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }
    return names.get(s, int(float(s)))

# ----------------------- weibull io -----------------------

def _load_weibull_monthly(csv_path: str, city: str) -> List[Tuple[float, float, float]]:
    """Return list (12×) of tuples: (k, c_at_zref, z_ref_m)."""
    df = pd.read_csv(csv_path)
    need = {"city", "month", "k", "c"}
    if not need.issubset(df.columns):
        raise ValueError(f"Weibull CSV must have columns {need}, got {list(df.columns)}")
    sub = df[df["city"].astype(str).str.strip().str.lower() == city.strip().lower()].copy()
    if sub.empty:
        raise ValueError(f"City '{city}' not found in {csv_path}")
    sub["m"] = sub["month"].apply(_std_month)
    sub = sub.sort_values("m")
    if len(sub) != 12:
        raise ValueError(f"Expected 12 rows for city={city}, got {len(sub)}")
    has_z = "z_ref" in sub.columns
    out: List[Tuple[float, float, float]] = []
    for _, r in sub.iterrows():
        out.append((float(r["k"]), float(r["c"]), float(r["z_ref"]) if has_z else 10.0))
    return out

# --------------------- power curve layer ------------------

@dataclass
class _Curve:
    v: np.ndarray        # m/s
    p_kw: np.ndarray     # kW
    rated_kw: float      # kW
    v_cut_in: float
    v_cut_out: float


def _iec_like_curve(rated_kw: float) -> _Curve:
    v = np.linspace(0.0, 30.0, 401)
    v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
    p = np.zeros_like(v)
    mask1 = (v >= v_cut_in) & (v < v_rated)
    p[mask1] = rated_kw * ((v[mask1] - v_cut_in) / (v_rated - v_cut_in)) ** 3
    p[(v >= v_rated) & (v < v_cut_out)] = rated_kw
    return _Curve(v=v, p_kw=p, rated_kw=rated_kw, v_cut_in=v_cut_in, v_cut_out=v_cut_out)


def _try_windpowerlib_curve(
    turbine_name: Optional[str],
    hub_height_m: Optional[float] = None,
    turbine_library_csv: Optional[str] = None,
) -> Optional[_Curve]:
    if not turbine_name:
        return None
    try:
        from windpowerlib import wind_turbine, get_turbine_types
        turb_lib = None
        if turbine_library_csv and os.path.exists(turbine_library_csv):
            turb_lib = pd.read_csv(turbine_library_csv)

        # --- 1) direct attempt with the provided name (exact turbine_type)
        try:
            wt = wind_turbine.WindTurbine(
                turbine_type=str(turbine_name),
                hub_height=float(hub_height_m) if hub_height_m is not None else 100.0,
                turbine_library=turb_lib,
            )
            pc = getattr(wt, "power_curve", None)
            if pc is not None and "wind_speed" in pc.columns:
                v = pd.to_numeric(pc["wind_speed"], errors="coerce").to_numpy(dtype=float)
                # power|value like your test
                if "power" in pc.columns:
                    p_series = pd.to_numeric(pc["power"], errors="coerce")
                elif "value" in pc.columns:
                    p_series = pd.to_numeric(pc["value"], errors="coerce")
                else:
                    num_cols = [c for c in pc.columns if c != "wind_speed" and pd.api.types.is_numeric_dtype(pc[c])]
                    if not num_cols:
                        p_series = None
                    else:
                        p_series = pd.to_numeric(pc[num_cols[0]], errors="coerce")
                if p_series is not None:
                    P = p_series.to_numpy(dtype=float) / 1000.0  # W→kW
                    good = np.isfinite(v) & np.isfinite(P)
                    v, P = v[good], P[good]
                    if v.size:
                        order = np.argsort(v)
                        v, P = v[order], P[order]
                        rated_kw = float(np.nanmax(P)) if np.isfinite(P).any() else 0.0
                        if rated_kw > 0:
                            v_pos = v[P > 0]
                            v_cut_in = float(v_pos.min()) if v_pos.size else 3.0
                            v_cut_out = float(v.max())
                            return _Curve(v=v, p_kw=P, rated_kw=rated_kw,
                                          v_cut_in=v_cut_in, v_cut_out=v_cut_out)
        except Exception:
            pass  # fall through to search

        # --- 2) fallback: search list and pick exact/contains match
        try:
            avail = get_turbine_types(turbine_library=turb_lib)
        except Exception:
            avail = wind_turbine.get_turbine_types(turbine_library=turb_lib)

        pairs: List[Tuple[str, str]] = []
        if isinstance(avail, pd.DataFrame) and {"manufacturer","turbine_type"}.issubset(avail.columns):
            for _, row in avail.iterrows():
                pairs.append((str(row["manufacturer"]), str(row["turbine_type"])))
        else:
            idx = getattr(avail, "index", None)
            if idx is not None:
                for it in idx.tolist():
                    if isinstance(it, tuple) and len(it) >= 2:
                        pairs.append((str(it[0]), str(it[1])))

        q = str(turbine_name).strip().lower()
        chosen_type = None
        # prefer exact turbine_type first
        for man, typ in pairs:
            if q == typ.lower():
                chosen_type = typ; break
        if chosen_type is None:
            # then manufacturer+type contains
            for man, typ in pairs:
                if q in (man + " " + typ).lower():
                    chosen_type = typ; break
        if chosen_type is None:
            return None

        wt = wind_turbine.WindTurbine(
            turbine_type=chosen_type,
            hub_height=float(hub_height_m) if hub_height_m is not None else 100.0,
            turbine_library=turb_lib,
        )
        pc = getattr(wt, "power_curve", None)
        if pc is None or "wind_speed" not in pc.columns:
            return None
        v = pd.to_numeric(pc["wind_speed"], errors="coerce").to_numpy(dtype=float)
        if "power" in pc.columns:
            p_series = pd.to_numeric(pc["power"], errors="coerce")
        elif "value" in pc.columns:
            p_series = pd.to_numeric(pc["value"], errors="coerce")
        else:
            num_cols = [c for c in pc.columns if c != "wind_speed" and pd.api.types.is_numeric_dtype(pc[c])]
            if not num_cols: return None
            p_series = pd.to_numeric(pc[num_cols[0]], errors="coerce")
        P = p_series.to_numpy(dtype=float) / 1000.0
        good = np.isfinite(v) & np.isfinite(P)
        v, P = v[good], P[good]
        if not v.size: return None
        order = np.argsort(v); v, P = v[order], P[order]
        rated_kw = float(np.nanmax(P)) if np.isfinite(P).any() else 0.0
        if rated_kw <= 0: return None
        v_pos = v[P > 0]
        v_cut_in = float(v_pos.min()) if v_pos.size else 3.0
        v_cut_out = float(v.max())
        return _Curve(v=v, p_kw=P, rated_kw=rated_kw, v_cut_in=v_cut_in, v_cut_out=v_cut_out)
    except Exception:
        return None



def _p_of_v(v: np.ndarray, curve: _Curve) -> np.ndarray:
    return np.interp(v, curve.v, curve.p_kw, left=0.0, right=0.0)

# --------- calibration profile (Mode B helpers) -----------

try:
    import yaml as _yaml2
    _HAS_YAML2 = True
except Exception:
    _HAS_YAML2 = False


def _apply_calibration_to_curve(curve: _Curve, power_scale: float = 1.0, v_shift: float = 0.0) -> _Curve:
    """
    Apply SCADA calibration:
      P_cal(v) = min( power_scale * P_manuf(v + v_shift), rated_orig )
    """
    if abs(power_scale - 1.0) < 1e-12 and abs(v_shift) < 1e-12:
        return curve
    v_new = curve.v.copy()
    p_shifted = np.interp(v_new + float(v_shift), curve.v, curve.p_kw, left=0.0, right=0.0)
    p_scaled = p_shifted * float(power_scale)
    p_capped = np.minimum(p_scaled, curve.rated_kw)
    return _Curve(v=v_new, p_kw=p_capped, rated_kw=curve.rated_kw, v_cut_in=curve.v_cut_in, v_cut_out=curve.v_cut_out)


def _load_calibration_profile(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml") and _HAS_YAML2:
            return _yaml2.safe_load(f) or {}
        return json.load(f)

# --------------------- physics helpers --------------------

def _power_law_adjust(c_ref: float, z_ref: float, z_hub: float, alpha: float) -> float:
    return c_ref * (z_hub / z_ref) ** alpha


def _apply_ti_noise(v: np.ndarray, ti: float, rng: np.random.Generator) -> np.ndarray:
    if ti <= 0.0:
        return v
    noise = rng.normal(0.0, ti, size=v.shape)
    v_eff = v * (1.0 + noise)
    return np.clip(v_eff, 0.0, None)


def _sample_weibull_hours(k: float, c: float, hours: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, size=hours)
    return c * (-np.log1p(-u)) ** (1.0 / k)


def _ar1_on_log(v: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    if phi <= 0.0:
        return v
    x = np.log(v + 1e-9)
    eps_std = np.std(x) * 0.3 if np.std(x) > 0 else 0.1
    eps = rng.normal(0.0, eps_std, size=x.shape)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = phi * y[t-1] + (1 - phi) * x[t] + eps[t]
    return np.exp(y) - 1e-9


def _cutout_hysteresis(v: np.ndarray, v_cut_in: float, v_cut_out: float) -> np.ndarray:
    on = np.ones_like(v, dtype=bool)
    off = False
    for i, vi in enumerate(v):
        if off:
            on[i] = False
            if vi <= v_cut_in:
                off = False
                on[i] = True
        else:
            on[i] = True
            if vi >= v_cut_out:
                off = True
                on[i] = False
    return on

# --------------------- Mode C (diurnal) -------------------

def _ensure_ts_col(df: pd.DataFrame, timestamp_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if timestamp_col and timestamp_col in df.columns:
        col = timestamp_col
    else:
        col = None
        for c in ("timestamp", "time", "datetime", "date", "Unnamed: 0"):
            if c in df.columns:
                col = c
                break
        if col is None:
            raise ValueError("No timestamp column found. Provide forecasting.timestamp_col in config.")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df.dropna(subset=[col]).set_index(col).sort_index(), col


def _pick_value_col(df: pd.DataFrame, prefs: List[str]) -> str:
    for c in prefs:
        if c in df.columns:
            return c
    raise ValueError(f"None of preferred value columns found: {prefs}")


def learn_diurnal_profile_from_csv(
    csv_path: str,
    timestamp_col: Optional[str] = None,
    value_pref: Optional[List[str]] = None,
    by_month: bool = True,
    freq: str = "H",
    min_days_per_month: int = 5,
    smooth_hours: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a multiplicative diurnal profile from a forecasting/operational dataset.

    Returns
    -------
    profile : np.ndarray
        (12,24) if by_month=True; otherwise (24,).
        Mean is 1 (per-month if monthly profile).
    info : dict
        Metadata about learning.
    """
    df = pd.read_csv(csv_path)
    df, col = _ensure_ts_col(df, timestamp_col)
    prefs = value_pref or ["wind_speed", "ws", "v", "power", "p", "p_kw", "p_mw"]
    val_col = _pick_value_col(df, prefs)

    x = pd.to_numeric(df[val_col], errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    x = x.resample(freq).mean().dropna()

    if len(x) > 10:
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        x = x.clip(lo, hi)

    def _runmean_circ(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return arr.copy()
        pad = k // 2
        arrp = np.r_[arr[-pad:], arr, arr[:pad]]
        ker = np.ones(k) / k
        sm = np.convolve(arrp, ker, mode="valid")
        return sm

    if by_month:
        prof = np.zeros((12, 24), dtype=float)
        mask = np.zeros(12, dtype=bool)
        gh = x.groupby(x.index.hour).mean().reindex(range(24)).to_numpy()
        gh = _runmean_circ(gh, smooth_hours)
        gh = gh / (gh.mean() or 1.0)
        for mi in range(12):
            xm = x[x.index.month == (mi + 1)]
            if xm.index.normalize().nunique() < min_days_per_month:
                prof[mi] = gh
                continue
            h = xm.groupby(xm.index.hour).mean().reindex(range(24)).to_numpy()
            h = _runmean_circ(h, smooth_hours)
            mu = h.mean()
            prof[mi] = (h / mu) if mu > 0 else np.ones(24)
            mask[mi] = True
        if not mask.all():
            fallback = prof[mask].mean(axis=0) if mask.any() else np.ones(24)
            for mi in range(12):
                if not mask[mi]:
                    prof[mi] = fallback
        info = {"mode": "monthly_24h", "smooth_hours": smooth_hours, "by_month": True,
                "value_col": val_col, "freq": freq}
        return prof, info
    else:
        h = x.groupby(x.index.hour).mean().reindex(range(24)).to_numpy()
        h = _runmean_circ(h, smooth_hours)
        mu = h.mean()
        prof = (h / mu) if mu > 0 else np.ones(24)
        info = {"mode": "global_24h", "smooth_hours": smooth_hours, "by_month": False,
                "value_col": val_col, "freq": freq}
        return prof, info


def _apply_diurnal_to_speed(v_hourly: np.ndarray, month_idx0: int, diurnal: Optional[np.ndarray]) -> np.ndarray:
    if diurnal is None:
        return v_hourly
    hours = np.arange(v_hourly.size) % 24
    f = diurnal[month_idx0, hours] if diurnal.ndim == 2 else diurnal[hours]
    v_mod = v_hourly * f
    mu0, mu1 = v_hourly.mean(), v_mod.mean()
    if mu1 > 0 and mu0 > 0:
        v_mod = v_mod * (mu0 / mu1)
    return v_mod

# --------------------------- API --------------------------

def sample_production_paths_monthly(
    n_iter: int,
    months_h: int,
    rng: np.random.Generator,
    cfg_yaml: Dict
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Returns:
      energy_paths_mwh: (N, T) monthly energy
      cf_quantiles:     (3, T)  P10, P50, P90 of monthly CF (gross)
      meta:             dict
      charts:           dict (unused here)
    """
    # --- read config
    wind = cfg_yaml.get("wind", {}) or {}
    plant = cfg_yaml.get("plant", {}) or {}
    project = cfg_yaml.get("project", {}) or {}
    scada_cfg = cfg_yaml.get("scada", {}) or {}
    fcfg = cfg_yaml.get("forecasting", {}) or {}

    weib_csv = wind.get("weibull_csv_path") or wind.get("weibull_csv") \
               or r"C:\HDD\Backup\uni\Project\WindPowerPlantAnalysis\Datasets\weibull_monthly.csv"
    city = wind.get("city", "Khaf")
    z_hub = float(wind.get("hub_height_m", 100.0))
    alpha = float(wind.get("alpha", 0.14))
    ti = float(wind.get("turbulence_intensity", 0.0)) if "turbulence_intensity" in wind else None
    diurnal_basic = bool(wind.get("diurnal_enable", False))
    ar1_phi = float(wind.get("ar1_phi", 0.0))
    spm = int(wind.get("samples_per_month", 720))

    cap_mw = float(plant.get("capacity_mw", 20.0))
    avail = float(plant.get("availability", 0.97)) if "availability" in plant else None
    losses = float(plant.get("losses_fraction", 0.10))
    degr_y = float(plant.get("degradation_per_year", 0.0))
    degr_m = (1.0 - degr_y) ** (1.0 / 12.0) if degr_y != 0 else 1.0

    start_year = int(project.get("start_year", project.get("start_jalali_year", 2010)))
    start_month = int(project.get("start_month", project.get("start_jalali_month", 1)))

    mode = str(wind.get("mode", "weibull")).lower()

    # --- Weibull monthly for city
    weib_m = _load_weibull_monthly(weib_csv, city)
    months = _month_index(start_year, start_month, months_h)

    # --- turbine power curve (manufacturer or IEC-like)
    curve = None
    src = str(wind.get("power_curve_source", "auto")).lower()
    turbine_library_csv = wind.get("turbine_library_csv")
    force_iec = bool(wind.get("force_iec_like", False))

    if not force_iec and src in ("auto", "windpowerlib"):
        curve = _try_windpowerlib_curve(
            wind.get("turbine_name"),
            hub_height_m=z_hub,
            turbine_library_csv=turbine_library_csv,
        )
        if curve is not None:
            src = "windpowerlib"

    if curve is None:
        rated_kw_hint = float(wind.get("rated_kw")) if wind.get("rated_kw") else cap_mw * 1000.0
        curve = _iec_like_curve(rated_kw_hint)
        src = "iec_like"


    # --- Mode B: apply SCADA calibration if requested
    calib_meta: Dict[str, Any] = {}
    if mode == "scada_calibrated":
        calib_path = scada_cfg.get("calibration_profile_path")
        if calib_path and os.path.exists(calib_path):
            prof = _load_calibration_profile(calib_path)
            ps_orig = float(prof.get("power_scale", 1.0))
            power_scale = min(1.3, ps_orig)     # soft cap برای جلوگیری از اشباع غیرواقعی
            v_shift = float(prof.get("v_shift", 0.0))
            curve = _apply_calibration_to_curve(curve, power_scale=power_scale, v_shift=v_shift)
            if ps_orig != power_scale:
                prof = dict(prof)
                prof["power_scale_capped_at"] = 1.3
                prof["power_scale_before_cap"] = ps_orig
            if ti is None and ("ti_mean" in prof):
                ti = float(prof["ti_mean"])
            if avail is None and ("availability_obs" in prof):
                avail = float(prof["availability_obs"])
            calib_meta = {"calibration_profile_path": calib_path, "profile": prof}
        else:
            calib_meta = {"warning": "SCADA mode requested but calibration_profile_path not found; using manufacturer/IEC curve."}

    if ti is None:
        ti = 0.0
    if avail is None:
        avail = 0.97

    # --- Mode C: learn diurnal profile if enabled
    diurnal_profile: Optional[np.ndarray] = None
    diurnal_info: Dict[str, Any] = {"enabled": False}
    if mode in ("hybrid_forecast", "weibull", "scada_calibrated") and fcfg.get("enabled"):
        try:
            diurnal_profile, diurnal_info = learn_diurnal_profile_from_csv(
                csv_path=fcfg["file"],
                timestamp_col=fcfg.get("timestamp_col"),
                value_pref=fcfg.get("value_pref"),
                by_month=bool(fcfg.get("by_month", True)),
                freq=str(fcfg.get("freq", "H")).lower(),  # ← FutureWarning fix
                min_days_per_month=int(fcfg.get("min_days_per_month", 5)),
                smooth_hours=int(fcfg.get("smooth_hours", 3)),
            )
            diurnal_info["enabled"] = True
        except Exception as e:
            diurnal_profile = None
            diurnal_info = {"enabled": False, "warning": f"diurnal learning failed: {e}"}

    # --- simulate
    N, T = int(n_iter), int(months_h)
    energy_paths = np.zeros((N, T), dtype=float)
    cf_all = np.zeros((N, T), dtype=float)

    for t, (yy, mm) in enumerate(months):
        k_m, c_ref_m, z_ref_m = weib_m[(mm - 1) % 12]
        c_hub = _power_law_adjust(c_ref_m, z_ref_m, z_hub, alpha)

        H = _hours_in_month(yy, mm)
        hours = spm if spm > 0 else H

        # Weibull random hourly speeds
        U = rng.uniform(0.0, 1.0, size=(N, hours))
        v = c_hub * (-np.log1p(-U)) ** (1.0 / k_m)

        # Diurnal modulation
        if diurnal_profile is not None:
            for i in range(N):
                v[i] = _apply_diurnal_to_speed(v[i], (mm - 1) % 12, diurnal_profile)
        elif diurnal_basic:
            f = 1.0 + 0.10 * np.sin(2 * np.pi * (np.arange(hours) % 24) / 24.0 - np.pi / 2)
            vm = v.mean(axis=1, keepdims=True)
            v = v * f
            v = np.where(vm > 0, v * (vm / (v.mean(axis=1, keepdims=True) + 1e-9)), v)

        # AR(1) on log-speed
        if ar1_phi > 0.0:
            for i in range(N):
                v[i] = _ar1_on_log(v[i], ar1_phi, rng)

        # TI noise
        if ti > 0.0:
            v = _apply_ti_noise(v, ti, rng)

        # Cut-in/out hysteresis → on/off mask
        on_mask = np.stack([_cutout_hysteresis(v[i], curve.v_cut_in, curve.v_cut_out) for i in range(N)], axis=0)
        v_eff = np.where(on_mask, v, 0.0)

        # Power mapping and monthly CF (gross)
        p_kw = _p_of_v(v_eff, curve)
        cf_month = np.clip((p_kw.mean(axis=1)) / max(1e-9, curve.rated_kw), 0.0, 1.0)
        cf_all[:, t] = cf_month

        # Monthly energy (net): availability, losses, degradation
        e_mwh = cf_month * cap_mw * H
        e_mwh *= avail * (1.0 - losses) * (degr_m ** t)

        energy_paths[:, t] = e_mwh

    cf_q = np.quantile(cf_all, [0.10, 0.50, 0.90], axis=0)

    meta: Dict[str, Any] = {
        "mode": mode,
        "city": city,
        "weibull_csv": weib_csv,
        "hub_height_m": z_hub,
        "alpha": alpha,
        "turbulence_intensity": ti,
        "diurnal": diurnal_info,
        "ar1_phi": ar1_phi,
        "samples_per_month": spm,
        "rated_kw": curve.rated_kw,
        "availability": avail,
        "losses_fraction": losses,
        "degradation_yearly": degr_y,
        "power_curve_source": src,
        "turbine_name": wind.get("turbine_name", None),
        "capacity_mw": cap_mw,
        "calibration": calib_meta,
    }
    charts: Dict[str, Any] = {}

    return energy_paths, cf_q, meta, charts


# ------------------------- main ---------------------------

if __name__ == "__main__":
    import yaml

    CONFIG_PATH = r"C:\HDD\Backup\uni\Project\WindPowerPlantAnalysis\Code\config.yaml"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project = cfg.get("project", {}) or {}
    years = int(project.get("years", 10))
    start_year = int(project.get("start_year", project.get("start_jalali_year", 2010)))
    start_month = int(project.get("start_month", project.get("start_jalali_month", 1)))

    T = years * 12
    rng = np.random.default_rng(42)

    energy_paths, cf_q, meta, _ = sample_production_paths_monthly(
        n_iter=100, months_h=T, rng=rng, cfg_yaml=cfg
    )

    out_dir = os.path.join(".", "Outputs")
    os.makedirs(out_dir, exist_ok=True)
    months = _month_index(start_year, start_month, T)
    df = pd.DataFrame({
        "Year": [y for (y, m) in months],
        "Month": [m for (y, m) in months],
        "Energy_MWh_P50": energy_paths.mean(axis=0),
        "Energy_MWh_P10": np.quantile(energy_paths, 0.10, axis=0),
        "Energy_MWh_P90": np.quantile(energy_paths, 0.90, axis=0),
        "CF_P10": cf_q[0],
        "CF_P50": cf_q[1],
        "CF_P90": cf_q[2],
    })
    out_path = os.path.join(out_dir, "production_paths_preview.csv")
    df.to_csv(out_path, index=False)

    print("[ok] Production model ran.")
    print(f"  Mode: {meta['mode']}")
    print(f"  City: {meta['city']}, Turbine: {meta.get('turbine_name') or 'IEC-like'}")
    print(f"  Rated (kW): {meta['rated_kw']:.0f}, Capacity (MW): {meta['capacity_mw']}")
    print(f"  TI: {meta['turbulence_intensity']}, AR1: {meta['ar1_phi']}")
    if meta.get("calibration"):
        print(f"  Calibration: {meta['calibration']}")
    if isinstance(meta.get('diurnal'), dict):
        print(f"  Diurnal: {meta['diurnal']}")
    print(f"  Paths shape: {energy_paths.shape}  (N,T)")
    print(f"  Preview CSV: {os.path.abspath(out_path)}")
