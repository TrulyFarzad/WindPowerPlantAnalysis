# coding: utf-8
"""
SCADA-calibration (Mode B) — سبک و بدون وابستگی به OpenOA:
- ورودی: لیست CSVهای SCADA (ستون‌های نام‌گذاری انعطاف‌پذیر)
- خروجی: calibration_profile (dict) + ذخیره به JSON/YAML
- کاری که می‌کند:
  * میانگین‌گیری بینه‌ای از Power vs WindSpeed → "effective" power curve
  * تخمین TI از درونِ بین‌ها (std/mean)
  * تخمین availability از نسبت نمونه‌های معتبر
  * تخمین scale/shift نسبت به منحنی سازنده‌ی windpowerlib
  * تخمین cut-in/out (دلتا) در صورت نیاز (این نسخه، scale/shift را پیاده می‌کند)
"""

from __future__ import annotations
import json, os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import yaml as _yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ---------- utils: column detection ----------
_WS_COLS = ["wind_speed", "ws", "wind", "v", "windspeed", "wind_speed_ms", "wind (m/s)"]
_P_COLS  = ["power_kw", "p_kw", "p", "power", "power(w)", "power_w"]

def _pick_col(cols, candidates):
    s = {c.lower().strip(): c for c in cols}
    for name in candidates:
        if name in s: return s[name]
    # fuzzy
    for c in cols:
        lc = c.lower().strip()
        for name in candidates:
            if name in lc:
                return c
    raise ValueError(f"Cannot find any of columns {candidates} in {list(cols)}")

def _to_kw(x: np.ndarray) -> np.ndarray:
    # اگر به نظر وات بود → به kW
    if np.nanmax(x) > 5_000:  # heuristics
        return x / 1000.0
    return x

def _smooth_ma(y: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1: return y
    k = min(win, max(1, len(y)//2*2+1))  # odd
    pad = k//2
    yy = np.pad(y, (pad, pad), mode='edge')
    c = np.ones(k)/k
    z = np.convolve(yy, c, mode='valid')
    return z

# ---------- core ----------
def fit_effective_power_curve_from_scada(
    scada_csvs: List[str],
    v_bin_edges: np.ndarray = np.arange(0.0, 30.0+0.5, 0.5),
    min_samples_per_bin: int = 20,
    smooth_window: int = 5
) -> Dict[str, np.ndarray]:
    frames = []
    for p in scada_csvs:
        df = pd.read_csv(p)
        ws = df[_pick_col(df.columns, _WS_COLS)].to_numpy(dtype=float)
        pw = df[_pick_col(df.columns, _P_COLS)].to_numpy(dtype=float)
        pw = _to_kw(pw)  # kW
        frames.append(pd.DataFrame({"ws": ws, "p_kw": pw}))
    sc = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()

    # اعتبار: مقادیر نامعقول را فیلتر کن
    sc = sc[(sc["ws"] >= 0.0) & (sc["ws"] <= 40.0)]
    sc = sc[(sc["p_kw"] >= 0.0) & (sc["p_kw"] <= sc["p_kw"].quantile(0.999))]

    # بین‌بندی
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

    # هموارسازی ساده
    m = np.isfinite(p_mean); 
    p_mean_s = p_mean.copy()
    if m.sum() >= 3:
        p_mean_s[m] = _smooth_ma(p_mean[m], smooth_window)

    # TI ~ E[ std(ws)/mean(ws) ] در بین‌های معتبر
    ti_vals = []
    for i in range(nb):
        if np.isfinite(ws_mean[i]) and ws_mean[i] > 0 and np.isfinite(ws_std[i]):
            ti_vals.append(ws_std[i]/ws_mean[i])
    ti_hat = float(np.nanmedian(ti_vals)) if ti_vals else 0.1

    return {
        "v_bin_edges": v_bin_edges,
        "v_mid": v_mid,
        "p_eff_kw": p_mean_s,
        "n_in_bin": n_in,
        "ti_mean": ti_hat
    }

def estimate_availability(scada_csvs: List[str], power_threshold_kw: float = 0.05) -> float:
    total = 0
    ok = 0
    for p in scada_csvs:
        df = pd.read_csv(p)
        pw = df[_pick_col(df.columns, _P_COLS)].to_numpy(dtype=float)
        pw = _to_kw(pw)
        total += len(pw)
        ok += int((pw >= power_threshold_kw).sum())
    if total == 0: return 0.95
    return float(ok/total)

def compare_with_manufacturer_curve(
    v_mid: np.ndarray,
    p_eff_kw: np.ndarray,
    manuf_v: np.ndarray,
    manuf_p_kw: np.ndarray,
    v_range: Tuple[float,float] = (4.0, 12.0)
) -> Tuple[float, float]:
    """
    خروجی:
      power_scale ≈ median(P_eff / P_manuf) روی بازه‌ی تعریف‌شده
      v_shift     ≈ شیفت افقی (m/s) برای کمینه کردن SSE ساده
    """
    # ماسک روی بازه
    mask = (v_mid >= v_range[0]) & (v_mid <= v_range[1]) & np.isfinite(p_eff_kw)
    if not np.any(mask) or manuf_v.size == 0:
        return 1.0, 0.0

    # مقادیر متناظر از منحنی سازنده
    p_man_on_mid = np.interp(v_mid[mask], manuf_v, manuf_p_kw, left=np.nan, right=np.nan)
    m2 = np.isfinite(p_man_on_mid) & np.isfinite(p_eff_kw[mask])
    if not np.any(m2): 
        return 1.0, 0.0

    ratio = p_eff_kw[mask][m2] / np.clip(p_man_on_mid[m2], 1e-6, None)
    power_scale = float(np.median(ratio))

    # شیفت افقی با جستجوی خطی ساده
    shifts = np.linspace(-1.5, 1.5, 31)
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

def build_calibration_profile(
    scada_csvs: List[str],
    manuf_curve: Tuple[np.ndarray, np.ndarray],  # (v, p_kw) از windpowerlib
    save_path: Optional[str] = None
) -> Dict:
    eff = fit_effective_power_curve_from_scada(scada_csvs)
    power_scale, v_shift = compare_with_manufacturer_curve(
        eff["v_mid"], eff["p_eff_kw"], manuf_curve[0], manuf_curve[1]
    )
    avail = estimate_availability(scada_csvs)

    profile = {
        "version": 1,
        "power_scale": power_scale,   # ضرب روی P_manuf
        "v_shift": v_shift,           # شیفت روی v در نگاشت
        "ti_mean": eff["ti_mean"],    # اگر در config مقداردهی نشده بود
        "availability_obs": avail,    # اگر خواستی به plant.availability تزریق کنی
        "notes": "Derived from SCADA by bin-averaging and simple alignment."
    }
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in (".yaml", ".yml") and _HAS_YAML:
            with open(save_path, "w", encoding="utf-8") as f:
                _yaml.safe_dump(profile, f, allow_unicode=True, sort_keys=False)
        else:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile
