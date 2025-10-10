
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build weibull_monthly.csv from Open-Meteo hourly wind data (2004-01-01 .. 2024-12-31).
- Source: Open-Meteo Historical Weather API (ERA5/ERA5-Land), hourly wind_speed_10m.
- Output schema: city, month, count, k, c, mean_ws, std_ws, status, file, location_id

Notes:
- Weibull fit uses method-of-moments: k ≈ (σ/μ)^(-1.086), c = μ / Γ(1 + 1/k)
  See: Justus et al. (1978); review in Appl. Meteor. (1984).
- For numerical stability, rows with μ<=0 or σ<=0 are flagged (status=LOW_WS) and k,c set NaN.
- Timezone is UTC; monthly grouping by calendar month.
- The script automatically chunks requests by year to avoid overly long URLs.
"""
import sys, time, math, csv, os, argparse
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from math import gamma

API = "https://archive-api.open-meteo.com/v1/archive"

def weibull_k_mom(mean, std):
    # Approximation by Justus et al. (1978)
    if mean <= 0 or std <= 0:
        return np.nan
    return (std / mean) ** (-1.086)

def weibull_c_from_mean(mean, k):
    if not np.isfinite(k) or k <= 0 or mean <= 0:
        return np.nan
    return mean / gamma(1.0 + 1.0/k)

def fetch_hourly(lat, lon, start, end, unit="ms", session=None, pause=0.8):
    """Fetch hourly wind_speed_10m between [start, end] inclusive (YYYY-MM-DD)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m",
        "wind_speed_unit": unit,
        "timeformat": "iso8601",
        "timezone": "UTC",
    }
    s = session or requests.Session()
    r = s.get(API, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    # Shape: {"hourly":{"time":[...], "wind_speed_10m":[...]}, ...}
    h = j.get("hourly", {})
    t = h.get("time", [])
    w = h.get("wind_speed_10m", [])
    if not t or not w or len(t)!=len(w):
        return pd.DataFrame(columns=["time","wind_speed_10m"])
    df = pd.DataFrame({"time": pd.to_datetime(t), "wind_speed_10m": pd.to_numeric(w, errors="coerce")})
    if pause>0:
        time.sleep(pause)
    return df

def yearly_chunks(y0=2004, y1=2024):
    for y in range(y0, y1+1):
        yield f"{y}-01-01", f"{y}-12-31"

def build_for_city(city, lat, lon, y0=2004, y1=2024):
    with requests.Session() as s:
        frames = []
        for sd, ed in yearly_chunks(y0, y1):
            df = fetch_hourly(lat, lon, sd, ed, session=s)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["wind_speed_10m"]).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()
    data["month"] = data["time"].dt.month
    # Aggregate by month across all years
    g = data.groupby("month")["wind_speed_10m"]
    out = g.agg(mean_ws="mean", std_ws="std", count="size").reset_index()
    # Weibull moments
    ks, cs, status = [], [], []
    for _, row in out.iterrows():
        mu, sd = float(row["mean_ws"]), float(row["std_ws"])
        if mu<=0 or sd<=0:
            ks.append(np.nan); cs.append(np.nan); status.append("LOW_WS")
        else:
            k = weibull_k_mom(mu, sd)
            c = weibull_c_from_mean(mu, k)
            ks.append(k); cs.append(c); status.append("OK")
    out["k"] = ks; out["c"] = cs; out["status"] = status
    out["city"] = city
    out["file"] = f"{lat:.4f}_{lon:.4f}.csv"
    out["location_id"] = "single"
    return out[["city","month","count","k","c","mean_ws","std_ws","status","file","location_id"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities_csv", default="cities_iran_10.csv", help="name,lat,lon")
    ap.add_argument("--start_year", type=int, default=2004)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--out_csv", default="weibull_monthly.csv")
    args = ap.parse_args()

    cities = pd.read_csv(args.cities_csv)
    rows = []
    for _, r in cities.iterrows():
        city, lat, lon = str(r["city"]), float(r["lat"]), float(r["lon"])
        print(f"[+] {city} ({lat:.4f},{lon:.4f})", flush=True)
        df = build_for_city(city, lat, lon, args.start_year, args.end_year)
        if df.empty:
            print(f"[!] No data for {city}", flush=True)
            continue
        rows.append(df)
    if not rows:
        print("No rows produced.")
        sys.exit(2)
    out = pd.concat(rows, ignore_index=True).sort_values(["city","month"])
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"[ok] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
