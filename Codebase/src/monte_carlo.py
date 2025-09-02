# -*- coding: utf-8 -*-
"""
monte_carlo.py — Post-processing on Monte Carlo outputs

Builds/uses price & energy paths (same RNG seed as other modules), then:
  - Computes revenue & cash paths
  - NPV / IRR distributions and summary
  - Payback period (discounted) distribution
  - Tornado sensitivity (CAPEX ±20%, OPEX ±20%, Discount ±2pp, Price ±20%, CF ±10%)
  - Scenario matrix (Price×CF multipliers) heatmap on NPV P50
  - Cumulative cash fan chart

Outputs (saved next to config.yaml):
  - mc_npv_distribution.csv
  - mc_irr_distribution.csv
  - mc_payback_distribution.csv
  - mc_scenario_matrix.csv
  - mc_tornado.png
  - mc_scenario_heatmap.png
  - mc_payback_hist.png
  - mc_cumcash_fan.png
"""

# --- imports (top of src/monte_carlo.py) ---
from __future__ import annotations
import os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Robust intra-package imports: try package-relative first, fallback to local
try:
    from .cashflow import (
        _load_yaml_config,
        _monthly_rate,
        _npv_monthly,
        _irr_monthly,
        build_monthly_vectors,
        build_cashflows,
        evaluate_paths,
    )
    from .wind_resource import _jalali_month_range  # for consistent date axis
except ImportError:
    from cashflow import (
        _load_yaml_config,
        _monthly_rate,
        _npv_monthly,
        _irr_monthly,
        build_monthly_vectors,
        build_cashflows,
        evaluate_paths,
    )
    from wind_resource import _jalali_month_range

# ----------------------------- helpers -----------------------------

def _jalali_month_index(start_y: int, start_m: int, end_y: int, end_m: int) -> pd.DatetimeIndex:
    """
    Gregorian month-end index from a Jalali range.
    If jdatetime is not available, fall back to a  month-end range of the same length.
    (Uses 'ME' to avoid pandas 'M' deprecation.)
    """
    months = (end_y - start_y) * 12 + (end_m - start_m) + 1
    try:
        import jdatetime
        dates = []
        y, m = start_y, start_m
        for _ in range(months):
            g = jdatetime.date(y, m, 1).togregorian()
            dates.append(pd.Timestamp(g.year, g.month, 1) + pd.offsets.MonthEnd(0))
            m = 1 if m == 12 else (m + 1)
            if m == 1:
                y += 1
        return pd.DatetimeIndex(dates)
    except Exception:
        start = pd.Timestamp.today().normalize().replace(day=1)
        return pd.date_range(start=start, periods=months, freq="ME")

def _baseline_run(cfg: Dict, start_jy=1404, start_jm=1, end_jy=1414, end_jm=12):
    """Return all baseline arrays needed for post-processing."""
    horizon_months = (end_jy - start_jy) * 12 + (end_jm - start_jm) + 1
    x_dates = _jalali_month_index(start_jy, start_jm, end_jy, end_jm)

    price_paths, energy_paths, opex_monthly = build_monthly_vectors(cfg, horizon_months)
    cash_paths, revenue_paths = build_cashflows(cfg, price_paths, energy_paths, opex_monthly)

    ann_disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    r_m = _monthly_rate(ann_disc)
    N = cash_paths.shape[0]
    npvs = np.array([_npv_monthly(cash_paths[i], r_m) for i in range(N)], dtype=float)
    irr_m = np.array([_irr_monthly(cash_paths[i], guess=0.01) for i in range(N)], dtype=float)
    irr_a = (1.0 + irr_m) ** 12.0 - 1.0

    return {
        "horizon_months": horizon_months,
        "x_dates": x_dates,
        "price_paths": price_paths,        # (N,T)
        "energy_paths": energy_paths,      # (N,T)
        "opex_monthly": opex_monthly,      # (T,)
        "cash_paths": cash_paths,          # (N,T)
        "revenue_paths": revenue_paths,    # (N,T)
        "npvs": npvs,                      # (N,)
        "irr_annual": irr_a,               # (N,)
        "r_m": r_m,
    }

def _discounted_payback_months(cash_paths: np.ndarray, r_m: float) -> np.ndarray:
    """
    Compute discounted payback period (in months) for each scenario (row).
    Returns array shape (N,) with month index (0-based), or np.nan if never pays back.
    """
    N, T = cash_paths.shape
    t = np.arange(T, dtype=float)
    disc = (1.0 + r_m) ** t
    dpb = np.full(N, np.nan, dtype=float)

    for i in range(N):
        cf = cash_paths[i] / disc
        csum = np.cumsum(cf)
        first_pos = np.where(csum > 0.0)[0]
        if first_pos.size > 0:
            dpb[i] = float(first_pos[0])
    return dpb

def _percentiles(arr: np.ndarray, qs: List[float]) -> np.ndarray:
    return np.percentile(arr, qs, axis=0)

# ----------------------------- Tornado sensitivity -----------------------------

def _npv_from_components(
    cfg: Dict,
    price_paths: np.ndarray,
    energy_paths: np.ndarray,
    opex_monthly: np.ndarray,
    capex_irr: float,
    discount_annual: float,
) -> np.ndarray:
    """
    Recompute NPV distribution given adjusted components.
    """
    N, T = price_paths.shape
    r_m = _monthly_rate(discount_annual)
    revenue_paths = energy_paths * price_paths
    opex_b = np.broadcast_to(opex_monthly.reshape(1, T), (N, T))
    cash_paths = revenue_paths - opex_b
    cash_paths[:, 0] -= capex_irr
    npvs = np.array([_npv_monthly(cash_paths[i], r_m) for i in range(N)], dtype=float)
    return npvs

def tornado_analysis(cfg: Dict, baseline: Dict) -> pd.DataFrame:
    """
    Compute delta in NPV P50 for ± changes in key drivers:
      - CAPEX ±20%
      - OPEX ±20%
      - Discount rate ±2 percentage points
      - Price level ±20%
      - CF level ±10%
    Returns a DataFrame with columns [factor, minus, plus, base_p50, delta_minus, delta_plus]
    """
    proj = cfg.get("project", {})
    costs = cfg.get("costs", {})
    plant = cfg.get("plant", {})

    cap_mw = float(plant.get("capacity_mw", 20.0))
    irr_per_usd = float(proj.get("irr_per_usd", 1.0))
    capex_usd_kw = float(costs.get("capex_usd_per_kw", 1000.0))
    capex_base = cap_mw * 1000.0 * capex_usd_kw * irr_per_usd
    disc_base = float(proj.get("discount_rate", 0.12))

    base_p50 = float(np.percentile(baseline["npvs"], 50))

    price0 = baseline["price_paths"].copy()
    energy0 = baseline["energy_paths"].copy()
    opex0 = baseline["opex_monthly"].copy()

    rows = []

    # 1) CAPEX ±20%
    for mult in [0.8, 1.2]:
        npvs = _npv_from_components(cfg, price0, energy0, opex0, capex_base * mult, disc_base)
        rows.append(("CAPEX", mult, float(np.percentile(npvs, 50))))
    # 2) OPEX ±20%
    for mult in [0.8, 1.2]:
        npvs = _npv_from_components(cfg, price0, energy0, opex0 * mult, capex_base, disc_base)
        rows.append(("OPEX", mult, float(np.percentile(npvs, 50))))
    # 3) Discount ±2pp
    for d in [-0.02, 0.02]:
        disc = max(1e-6, disc_base + d)
        npvs = _npv_from_components(cfg, price0, energy0, opex0, capex_base, disc)
        rows.append(("Discount Rate", disc, float(np.percentile(npvs, 50))))
    # 4) Price level ±20%
    for mult in [0.8, 1.2]:
        npvs = _npv_from_components(cfg, price0 * mult, energy0, opex0, capex_base, disc_base)
        rows.append(("Price Level", mult, float(np.percentile(npvs, 50))))
    # 5) CF level ±10%
    for mult in [0.9, 1.1]:
        npvs = _npv_from_components(cfg, price0, energy0 * mult, opex0, capex_base, disc_base)
        rows.append(("Capacity Factor", mult, float(np.percentile(npvs, 50))))

    df = pd.DataFrame(rows, columns=["Factor", "Variant", "NPV_P50"])
    out = []
    for fac, group in df.groupby("Factor"):
        g = group.sort_values("Variant")
        minus_val = g.iloc[0]["Variant"]
        plus_val = g.iloc[-1]["Variant"]
        minus_p50 = g.iloc[0]["NPV_P50"]
        plus_p50 = g.iloc[-1]["NPV_P50"]
        base_p50_fac = base_p50
        out.append({
            "factor": fac,
            "minus": minus_val,
            "plus": plus_val,
            "base_p50": base_p50_fac,
            "delta_minus": float(minus_p50 - base_p50_fac),
            "delta_plus": float(plus_p50 - base_p50_fac),
        })
    out_df = pd.DataFrame(out).sort_values("factor", ascending=True)
    return out_df

# ----------------------------- Scenario matrix -----------------------------

def scenario_matrix(cfg: Dict, baseline: Dict,
                    price_mults=(0.8, 0.9, 1.0, 1.1, 1.2),
                    cf_mults=(0.9, 1.0, 1.1)) -> pd.DataFrame:
    """
    Compute a matrix of NPV P50 for grid of (price_mult, cf_mult).
    """
    proj = cfg.get("project", {})
    costs = cfg.get("costs", {})
    plant = cfg.get("plant", {})

    cap_mw = float(plant.get("capacity_mw", 20.0))
    irr_per_usd = float(proj.get("irr_per_usd", 1.0))
    capex_usd_kw = float(costs.get("capex_usd_per_kw", 1000.0))
    capex_base = cap_mw * 1000.0 * capex_usd_kw * irr_per_usd
    disc_base = float(proj.get("discount_rate", 0.12))

    price0 = baseline["price_paths"]
    energy0 = baseline["energy_paths"]
    opex0 = baseline["opex_monthly"]

    rows = []
    for pm in price_mults:
        for cm in cf_mults:
            npvs = _npv_from_components(cfg, price0 * pm, energy0 * cm, opex0, capex_base, disc_base)
            p50 = float(np.percentile(npvs, 50))
            rows.append((pm, cm, p50))
    df = pd.DataFrame(rows, columns=["price_mult", "cf_mult", "NPV_P50"])
    return df

# ----------------------------- __main__ -----------------------------

if __name__ == "__main__":
    CODEBASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(CODEBASE, "config.yaml")
    cfg = _load_yaml_config(CONFIG_PATH)

    # Horizon identical to other scripts (Jalali 1404/01 → 1414/12)
    start_jy, start_jm = 1404, 1
    end_jy, end_jm = 1414, 12
    baseline = _baseline_run(cfg, start_jy, start_jm, end_jy, end_jm)

    # -------- Distributions CSVs --------
    pd.DataFrame({"NPV_IRR": baseline["npvs"]}).to_csv(
        os.path.join(CODEBASE, "mc_npv_distribution.csv"), index=False
    )
    irr_a = baseline["irr_annual"]
    pd.DataFrame({"IRR_annual": irr_a}).to_csv(
        os.path.join(CODEBASE, "mc_irr_distribution.csv"), index=False
    )

    # Payback distribution
    dpb = _discounted_payback_months(baseline["cash_paths"], baseline["r_m"])
    pd.DataFrame({"payback_months_discounted": dpb}).to_csv(
        os.path.join(CODEBASE, "mc_payback_distribution.csv"), index=False
    )

    # -------- Tornado analysis --------
    tor = tornado_analysis(cfg, baseline)
    tor.to_csv(os.path.join(CODEBASE, "mc_tornado.csv"), index=False)

    # Tornado plot
    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(tor))
    # draw minus (left) and plus (right) deltas
    plt.barh(y_pos, tor["delta_minus"], align="center")
    plt.barh(y_pos, tor["delta_plus"], align="center")
    plt.yticks(y_pos, tor["factor"].tolist())
    plt.axvline(0.0, color="k", linewidth=1)
    plt.title("Tornado: Δ NPV P50 by parameter shocks")
    plt.xlabel("Δ NPV P50 (IRR)")
    plt.ylabel("Parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(CODEBASE, "mc_tornado.png"), dpi=150)

    # -------- Scenario matrix (Price × CF) --------
    scen_df = scenario_matrix(cfg, baseline)
    scen_df.to_csv(os.path.join(CODEBASE, "mc_scenario_matrix.csv"), index=False)

    # Scenario heatmap
    piv = scen_df.pivot(index="cf_mult", columns="price_mult", values="NPV_P50")
    pm_vals = list(piv.columns)
    cm_vals = list(piv.index)
    plt.figure(figsize=(8, 5))
    plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.title("Scenario Matrix: NPV P50 (IRR) by Price×CF multipliers")
    plt.xlabel("Price multiplier")
    plt.ylabel("CF multiplier")
    plt.xticks(np.arange(len(pm_vals)), [str(v) for v in pm_vals])
    plt.yticks(np.arange(len(cm_vals)), [str(v) for v in cm_vals])
    plt.tight_layout()
    plt.savefig(os.path.join(CODEBASE, "mc_scenario_heatmap.png"), dpi=150)

    # -------- Payback histogram --------
    if np.isfinite(dpb).any():
        plt.figure(figsize=(9, 4))
        plt.hist(dpb[np.isfinite(dpb)], bins=50)
        plt.title("Discounted Payback Period Distribution")
        plt.xlabel("Months to payback")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(CODEBASE, "mc_payback_hist.png"), dpi=150)

    # -------- Cumulative cash fan (discounted) --------
    N, T = baseline["cash_paths"].shape
    t = np.arange(T, dtype=float)
    disc = (1.0 + baseline["r_m"]) ** t
    cumdisc = np.cumsum(baseline["cash_paths"] / disc.reshape(1, T), axis=1)
    qs = [5, 50, 95]
    qmat = _percentiles(cumdisc, qs)
    x_dates = baseline["x_dates"]

    plt.figure(figsize=(12, 4))
    plt.plot(x_dates, qmat[1], label="Cum. discounted cash P50")
    plt.fill_between(x_dates, qmat[0], qmat[2], alpha=0.2, label="P5–P95")
    plt.title("Cumulative Discounted Cash Flow — P50 & P5–P95")
    plt.xlabel("Month (Gregorian)")
    plt.ylabel("IRR (cumulative, discounted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CODEBASE, "mc_cumcash_fan.png"), dpi=150)

    print("Saved:",
          "mc_npv_distribution.csv, mc_irr_distribution.csv, mc_payback_distribution.csv,",
          "mc_tornado.csv, mc_scenario_matrix.csv,",
          "mc_tornado.png, mc_scenario_heatmap.png, mc_payback_hist.png, mc_cumcash_fan.png")
