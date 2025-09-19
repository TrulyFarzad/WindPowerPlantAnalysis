# coding: utf-8
# ==========================================================
# Build calibration_profile.json from SCADA CSVs — Robust++
# Mode B (SCADA-calibrated) — Standalone + Overrides + Counts
# ==========================================================

import os, json, math, warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# ============ USER SETTINGS (ABSOLUTE PATHS) ============
SCADA_FILES: List[str] = [
    r"C:\HDD\Backup\uni\Project\WindPowerPlantAnalysis\Datasets\wind turbine scada dataset\T1.csv",
]
TURBINE_NAME = "Vestas V112 3.0MW"   # اگر دقیق نبود، fallback استفاده می‌شود
RATED_KW_FALLBACK = 3000.0           # IEC-like وقتی windpowerlib در دسترس/یافت نشد
OUT_PATH = r"C:\HDD\Backup\uni\Project\WindPowerPlantAnalysis\Datasets\calibration_profile.json"

# ---- COLUMN OVERRIDES (نام دقیق ستون‌ها اگر می‌دانی) ----
# اگر خالی بگذاری، اسکریپت سعی می‌کند خودش پیدا کند.
FORCE_WS_COL: str = ""   # مثلا: "WindSpeed(m/s)" یا "V"
FORCE_P_COL:  str = ""   # مثلا: "Power(kW)" یا "P"

# ---- UNIT OVERRIDES ----
# "auto" | "m/s" | "km/h" | "knots"
FORCE_WS_UNIT: str = "auto"
# "auto" | "kW" | "MW" | "W" | "percent" | "counts4095"
FORCE_P_UNIT: str = "auto"

# ---- OPERATIONAL THRESHOLDS ----
CUT_IN_MS: float = 3.0                   # cut-in nominal برای availability
AVAIL_POWER_FRAC_OF_RATED: float = 0.10  # >10% rated => عملیاتی
V_SHIFT_RANGE: float = 6.0               # ±6 m/s
# ========================================================

# ---------- manufacturer curve (windpowerlib OR IEC-like) ----------
def get_manufacturer_curve(turbine_name: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from windpowerlib import wind_turbine
            wt = wind_turbine.WindTurbine(turbine_name)
            pc = wt.power_curve  # columns: wind_speed (m/s), power (W)
            v = pc["wind_speed"].to_numpy(dtype=float)
            P = pc["power"].to_numpy(dtype=float) / 1000.0  # → kW
            idx = np.argsort(v)
            v = v[idx]; P = P[idx]
            if not np.any(np.isfinite(P)) or np.nanmax(P) <= 0:
                raise RuntimeError("Empty/invalid windpowerlib curve")
            return v, P
    except Exception as e:
        raise RuntimeError(f"windpowerlib not available or turbine '{turbine_name}' not found: {e}")

def iec_like_curve(rated_kw: float) -> Tuple[np.ndarray, np.ndarray]:
    v = np.linspace(0.0, 30.0, 401)
    v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
    p = np.zeros_like(v)
    ramp = (v >= v_cut_in) & (v < v_rated)
    p[ramp] = rated_kw * ((v[ramp] - v_cut_in)/(v_rated - v_cut_in))**3
    p[(v >= v_rated) & (v < v_cut_out)] = rated_kw
    return v, p

# ---------- SCADA columns ----------
_WS_CANDIDATES = ["wind_speed", "ws", "wind", "v", "windspeed", "wind_speed_ms", "wind (m/s)"]
_P_CANDIDATES  = ["power_kw", "p_kw", "p", "power", "power(w)", "power_w", "power_mw", "power_percent"]

def _pick_col(df: pd.DataFrame, candidates: List[str], forced: str, taken: Optional[str]=None) -> str:
    cols = df.columns.tolist()
    if forced:
        if forced in df.columns:
            print(f"[pick] forced column: {forced}", flush=True)
            return forced
        else:
            raise ValueError(f"Forced column '{forced}' not in CSV columns: {cols}")

    # exact lowercase match first
    lower_map = {c.lower().strip(): c for c in cols}
    for name in candidates:
        if name in lower_map:
            c = lower_map[name]
            if taken is None or c != taken:
                print(f"[pick] exact candidate: {c}", flush=True)
                return c

    # fuzzy contains
    for c in cols:
        lc = c.lower().strip()
        for name in candidates:
            if name in lc and (taken is None or c != taken):
                print(f"[pick] fuzzy candidate: {c}", flush=True)
                return c
    raise ValueError(f"Cannot find any of columns {candidates} in {cols}")

def _dbg_stats(name: str, x: np.ndarray):
    q = np.nanpercentile(x, [0, 1, 5, 50, 95, 99, 100])
    print(f"[stats] {name}: min={q[0]:.4g} p1={q[1]:.4g} p5={q[2]:.4g} "
          f"median={q[3]:.4g} p95={q[4]:.4g} p99={q[5]:.4g} max={q[6]:.4g}", flush=True)

# ---------- unit handling ----------
def normalize_ws(ws: np.ndarray) -> Tuple[np.ndarray, str]:
    if FORCE_WS_UNIT != "auto":
        if FORCE_WS_UNIT == "m/s":   return ws, "m/s (forced)"
        if FORCE_WS_UNIT == "km/h":  return ws/3.6, "km/h -> m/s (forced)"
        if FORCE_WS_UNIT == "knots": return ws*0.514444, "knots -> m/s (forced)"
        raise ValueError("FORCE_WS_UNIT invalid")

    ws_mps = ws.astype(float)
    ws_med = np.nanmedian(ws_mps)
    if ws_med > 50:
        return ws_mps/3.6, "km/h -> m/s"
    elif 25 <= ws_med <= 50 and np.nanpercentile(ws_mps,95) > 60:
        return ws_mps*0.514444, "knots -> m/s"
    else:
        return ws_mps, "m/s (assumed)"

def normalize_power(pw: np.ndarray, rated_kw_hint: float) -> Tuple[np.ndarray, str]:
    # explicit force
    if FORCE_P_UNIT != "auto":
        if FORCE_P_UNIT == "kW":     return pw, "kW (forced)"
        if FORCE_P_UNIT == "MW":     return pw*1000.0, "MW -> kW (forced)"
        if FORCE_P_UNIT == "W":      return pw/1000.0, "W -> kW (forced)"
        if FORCE_P_UNIT == "percent":
            maxv = float(np.nanmax(pw)) if np.isfinite(pw).any() else 1.0
            if maxv <= 1.5:  # 0..1
                return pw * rated_kw_hint, "fraction -> kW (forced)"
            else:  # 0..100
                return (pw/100.0) * rated_kw_hint, "percent -> kW (forced)"
        if FORCE_P_UNIT == "counts4095":
            return (pw/4095.0) * rated_kw_hint, "counts4095 -> kW (forced)"
        raise ValueError("FORCE_P_UNIT invalid")

    # auto
    pw_kw = pw.astype(float)
    p_max = np.nanmax(pw_kw) if np.isfinite(pw_kw).any() else 0.0
    # counts heuristic
    if 2000.0 <= p_max <= 5000.0:
        return (pw_kw/4095.0)*rated_kw_hint, "counts4095 -> kW"
    if p_max > 50_000:            return pw_kw/1000.0, "W -> kW"
    if p_max < 50:                return pw_kw*1000.0, "MW -> kW"
    if 0.0 <= np.nanmedian(pw_kw) <= 1.0 and p_max <= 1.5:
        return pw_kw * rated_kw_hint, "fraction -> kW"
    if 1.5 < p_max <= 100 and np.nanmedian(pw_kw) <= 50:
        return (pw_kw/100.0) * rated_kw_hint, "percent -> kW"
    return pw_kw, "kW (assumed)"

def smooth_ma(y: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1: return y
    k = min(win, max(1, len(y)//2*2+1))  # odd
    pad = k//2
    yy = np.pad(y, (pad, pad), mode='edge')
    c = np.ones(k)/k
    z = np.convolve(yy, c, mode='valid')
    return z

def fit_effective_power_curve(csv_paths: List[str],
                              v_bin_edges: np.ndarray,
                              rated_kw_hint: float,
                              min_samples_per_bin: int = 20,
                              smooth_window: int = 5) -> Dict[str, np.ndarray]:
    frames = []
    unit_infos = []
    total_rows = 0
    for path in csv_paths:
        print(f"[load] {path}", flush=True)
        df = pd.read_csv(path)
        # pick columns (avoid picking same col twice)
        ws_col = _pick_col(df, _WS_CANDIDATES, FORCE_WS_COL, taken=None)
        p_col  = _pick_col(df, _P_CANDIDATES, FORCE_P_COL, taken=ws_col)
        print(f"[cols] ws_col='{ws_col}'   p_col='{p_col}'", flush=True)

        ws_raw = df[ws_col].to_numpy(dtype=float)
        pw_raw = df[p_col].to_numpy(dtype=float)
        _dbg_stats("ws_raw", ws_raw)
        _dbg_stats("pw_raw", pw_raw)

        # guard: if identical arrays, stop and ask for override
        if np.array_equal(ws_raw, pw_raw):
            raise RuntimeError("Selected WS and Power columns are identical. "
                               "Set FORCE_WS_COL and FORCE_P_COL to correct column names.")

        ws, ws_info = normalize_ws(ws_raw)
        pw, pw_info = normalize_power(pw_raw, rated_kw_hint)
        unit_infos.append({"ws_unit": ws_info, "p_unit": pw_info})
        frames.append(pd.DataFrame({"ws": ws, "p_kw": pw}))
        total_rows += len(df)
    sc = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[ok] rows loaded: {total_rows}  -> valid after dropna: {len(sc)}", flush=True)

    # sanity clip
    sc = sc[(sc["ws"] >= 0.0) & (sc["ws"] <= 40.0)]
    sc = sc[(sc["p_kw"] >= 0.0) & (sc["p_kw"] <= sc["p_kw"].quantile(0.999))]
    print(f"[ok] rows after plausibility clip: {len(sc)}", flush=True)

    # binning
    bidx = np.digitize(sc["ws"].to_numpy(), v_bin_edges) - 1
    nb = len(v_bin_edges) - 1
    v_mid = 0.5*(v_bin_edges[:-1] + v_bin_edges[1:])

    p_mean = np.full(nb, np.nan)
    p_std  = np.full(nb, np.nan)
    ws_mean = np.full(nb, np.nan)
    ws_std  = np.full(nb, np.nan)
    n_in = np.zeros(nb, dtype=int)

    for i in range(nb):
        sel = (bidx == i)
        if not np.any(sel): continue
        chunk = sc.loc[sel]
        n = len(chunk)
        if n < min_samples_per_bin: 
            continue
        n_in[i] = n
        p_mean[i] = chunk["p_kw"].mean()
        p_std[i]  = chunk["p_kw"].std()
        ws_mean[i]= chunk["ws"].mean()
        ws_std[i] = chunk["ws"].std()

    # smoothing
    m = np.isfinite(p_mean)
    p_mean_s = p_mean.copy()
    if m.sum() >= 3:
        p_mean_s[m] = smooth_ma(p_mean[m], smooth_window)

    # TI estimate
    ti_vals = []
    for i in range(nb):
        if np.isfinite(ws_mean[i]) and ws_mean[i] > 0 and np.isfinite(ws_std[i]):
            ti_vals.append(ws_std[i]/ws_mean[i])
    ti_hat = float(np.nanmedian(ti_vals)) if ti_vals else 0.1

    used_bins = int(np.sum(n_in >= min_samples_per_bin))
    print(f"[bins] used_bins={used_bins}/{nb}  median_TI≈{ti_hat:.3f}", flush=True)

    return {
        "v_bin_edges": v_bin_edges,
        "v_mid": v_mid,
        "p_eff_kw": p_mean_s,
        "n_in_bin": n_in,
        "ti_mean": ti_hat,
        "unit_infos": unit_infos
    }

def estimate_availability(csv_paths: List[str], rated_kw_hint: float,
                          cut_in_ms: float, p_frac: float) -> float:
    total = 0
    ok = 0
    p_thr_kw = p_frac * rated_kw_hint
    for path in csv_paths:
        df = pd.read_csv(path)
        ws_col = _pick_col(df, _WS_CANDIDATES, FORCE_WS_COL, taken=None)
        p_col  = _pick_col(df, _P_CANDIDATES, FORCE_P_COL, taken=ws_col)
        ws_raw = df[ws_col].to_numpy(dtype=float)
        pw_raw = df[p_col].to_numpy(dtype=float)
        ws, _ = normalize_ws(ws_raw)
        pw, _ = normalize_power(pw_raw, rated_kw_hint)
        mask = (ws >= cut_in_ms) & (pw >= p_thr_kw)
        total += len(pw)
        ok += int(mask.sum())
    if total == 0: return 0.95
    return float(ok/total)

def compare_with_manufacturer_curve(v_mid: np.ndarray,
                                    p_eff_kw: np.ndarray,
                                    manuf_v: np.ndarray,
                                    manuf_p_kw: np.ndarray,
                                    v_range: Tuple[float,float] = (4.0, 12.0),
                                    shift_range: float = 6.0) -> Tuple[float, float]:
    mask = (v_mid >= v_range[0]) & (v_mid <= v_range[1]) & np.isfinite(p_eff_kw)
    if not np.any(mask) or manuf_v.size == 0:
        return 1.0, 0.0

    p_man_on_mid = np.interp(v_mid[mask], manuf_v, manuf_p_kw, left=np.nan, right=np.nan)
    m2 = np.isfinite(p_man_on_mid) & np.isfinite(p_eff_kw[mask])
    if not np.any(m2): 
        return 1.0, 0.0

    ratio = p_eff_kw[mask][m2] / np.clip(p_man_on_mid[m2], 1e-6, None)
    r_med = np.median(ratio)
    r_mad = np.median(np.abs(ratio - r_med)) + 1e-6
    ok = (ratio >= r_med - 5*r_mad) & (ratio <= r_med + 5*r_mad)
    power_scale = float(np.median(ratio[ok])) if np.any(ok) else float(r_med)

    shifts = np.linspace(-shift_range, shift_range, int(shift_range*10)+1)  # step 0.1 m/s
    best_sse = np.inf
    best_shift = 0.0
    for s in shifts:
        p_man_s = np.interp(v_mid[mask], manuf_v + s, manuf_p_kw, left=np.nan, right=np.nan)
        mm = np.isfinite(p_man_s)
        if not np.any(mm): 
            continue
        sse = float(np.nanmean((p_eff_kw[mask][mm] - p_man_s[mm]*power_scale)**2))
        if sse < best_sse:
            best_sse = sse
            best_shift = float(s)
    return power_scale, best_shift

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    try:
        missing = [p for p in SCADA_FILES if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError("SCADA files not found:\n  - " + "\n  - ".join(missing))
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

        # manufacturer curve
        try:
            from math import isfinite
            v_man, p_man = get_manufacturer_curve(TURBINE_NAME)
            rated_kw = float(np.nanmax(p_man))
            print(f"[curve] windpowerlib curve loaded for '{TURBINE_NAME}': rated≈{rated_kw:.0f} kW", flush=True)
        except Exception as e:
            rated_kw = RATED_KW_FALLBACK
            v_man, p_man = iec_like_curve(rated_kw)
            print(f"[curve] Fallback IEC-like rated={rated_kw:.0f} kW ({e})", flush=True)

        # effective curve
        v_bins = np.arange(0.0, 30.0 + 0.5, 0.5)
        eff = fit_effective_power_curve(SCADA_FILES, v_bins, rated_kw)

        # availability (operational)
        avail = estimate_availability(SCADA_FILES, rated_kw, CUT_IN_MS, AVAIL_POWER_FRAC_OF_RATED)
        print(f"[avail] observed operational≈{avail:.3f} (ws≥{CUT_IN_MS} m/s & P≥{AVAIL_POWER_FRAC_OF_RATED*100:.0f}% rated)", flush=True)

        # alignment
        ps, vs = compare_with_manufacturer_curve(eff["v_mid"], eff["p_eff_kw"], v_man, p_man,
                                                 v_range=(max(3.0, CUT_IN_MS+0.5), 12.0),
                                                 shift_range=V_SHIFT_RANGE)
        print(f"[align] power_scale≈{ps:.4f}   v_shift≈{vs:.2f} m/s", flush=True)

        # profile
        profile = {
            "version": 4,
            "power_scale": float(ps),
            "v_shift": float(vs),
            "ti_mean": float(eff["ti_mean"]),
            "availability_obs": float(avail),
            "unit_infos": eff.get("unit_infos", []),
            "notes": "SCADA-derived with column overrides, counts handling, and operational availability."
        }
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        print(f"[ok] calibration_profile saved at:\n  {OUT_PATH}", flush=True)
        print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)

    except Exception as ex:
        print("[ERROR]", type(ex).__name__, ":", str(ex), flush=True)
        raise
