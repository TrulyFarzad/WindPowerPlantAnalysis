# prod/diurnal_forecast.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal

def _ensure_dt(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df.dropna(subset=[ts_col])

def _safe_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def learn_diurnal_profile(
    csv_path: str,
    timestamp_col: Optional[str] = None,
    value_pref: Tuple[str, ...] = ("wind_speed", "ws", "v", "power", "p", "p_kw", "p_mw"),
    freq: Literal["H","10min","15min","30min"] = "H",
    by_month: bool = True,
    min_days_per_month: int = 5,
    smooth_hours: int = 3,
    clip_quantiles: Tuple[float,float] = (0.01, 0.99),
) -> Tuple[np.ndarray, dict]:
    """
    بر اساس ستون سرعت باد یا توان (هر کدام موجود بود) یک پروفایل دیورنال می‌سازد.
    خروجی:
      - diurnal: shape = (12,24) اگر by_month=True، وگرنه (24,)
      - info: متادیتای یادگیری
    """
    df = pd.read_csv(csv_path)
    ts = timestamp_col or _safe_col(df, ("timestamp","time","datetime","date"))
    if ts is None:
        raise ValueError("No timestamp column found. Set `timestamp_col` explicitly.")
    df = _ensure_dt(df, ts)
    df = df.set_index(ts).sort_index()

    val_col = _safe_col(df, value_pref)
    if val_col is None:
        raise ValueError(f"No value column found among {value_pref}. Provide one.")

    # پاکسازی سبک
    x = df[val_col].astype(float).replace([np.inf, -np.inf], np.nan)
    lo, hi = x.quantile(clip_quantiles[0]), x.quantile(clip_quantiles[1])
    x = x.clip(lo, hi).interpolate(limit=6).dropna()
    # به فرکانس ساعتی نگاشت (میانگین/ریسمپل)
    x = x.resample(freq).mean().dropna()
    x = x[x > 0]  # از صفرهای مصنوعی پرهیز

    # ساخت ایندکس‌های کمکی
    h = x.index.tz_convert("UTC").hour  # ساعت روز 0..23
    m = x.index.tz_convert("UTC").month - 1  # 0..11

    def _month_profile(m_idx: int) -> Optional[np.ndarray]:
        xm = x[m == m_idx]
        if xm.index.normalize().nunique() < min_days_per_month:
            return None
        g = xm.groupby(xm.index.hour).mean()
        prof = g.reindex(range(24), fill_value=g.mean())
        prof = prof.to_numpy(dtype=float)
        prof = _smooth_running_mean(prof, k=smooth_hours)
        prof = prof / np.mean(prof) if np.mean(prof) > 0 else np.ones(24)
        return prof

    if by_month:
        diurnal = np.zeros((12, 24), dtype=float)
        mask = np.zeros(12, dtype=bool)
        for mi in range(12):
            p = _month_profile(mi)
            if p is not None:
                diurnal[mi] = p
                mask[mi] = True
        # اگر برای بعضی ماه‌ها داده ناکافی بود، از میانگین ماه‌های موجود استفاده کن
        if not mask.all():
            fallback = diurnal[mask].mean(axis=0) if mask.any() else np.ones(24)
            for mi in range(12):
                if not mask[mi]:
                    diurnal[mi] = fallback
        info = dict(mode="monthly_24h", smooth_hours=smooth_hours, by_month=True)
    else:
        g = x.groupby(h).mean().reindex(range(24), fill_value=x.mean())
        prof = g.to_numpy(dtype=float)
        prof = _smooth_running_mean(prof, k=smooth_hours)
        diurnal = prof / np.mean(prof) if np.mean(prof) > 0 else np.ones(24)
        info = dict(mode="global_24h", smooth_hours=smooth_hours, by_month=False)

    return diurnal, info

def _smooth_running_mean(y: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1: 
        return y.copy()
    pad = k//2
    yy = np.r_[y[-pad:], y, y[:pad]]  # wrap for circular hour
    ker = np.ones(k) / k
    sm = np.convolve(yy, ker, mode="valid")
    # convolve valid length = len(y) + pad*2 - k + 1 = len(y)
    return sm
