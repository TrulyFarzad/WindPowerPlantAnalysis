# -*- coding: utf-8 -*-
"""
economic_model.py  (a.k.a cashflow.py)

Wires together:
  - price_model.sample_price_path_monthly  -> price paths  [IRR/MWh]
  - wind_resource.sample_production_paths_monthly -> energy paths [MWh]

Builds monthly cash flows (Revenue - OPEX - CAPEX), then NPV/IRR distributions.
Saves plots & CSVs next to config.yaml.

Assumptions (MVP):
  * CAPEX paid at t=0 (first month) in IRR
  * OPEX is fixed per kW-year (from config), grows by annual inflation_opex
  * Discounting is monthly: r_m = (1+annual_discount)^(1/12)-1
  * No debt service/tax yet (kept for later iterations)
"""

# --- imports (top of cashflow.py) ---
from __future__ import annotations
import os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Robust intra-package imports: try package-relative first, fallback to local
try:
    from .price_model import _build_price_config, sample_price_path_monthly
    from .wind_resource import sample_production_paths_monthly, _jalali_month_range
except ImportError:
    from price_model import _build_price_config, sample_price_path_monthly
    from wind_resource import sample_production_paths_monthly, _jalali_month_range


# ----------------------------- utils -----------------------------

def _load_yaml_config(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _monthly_rate(annual_rate: float) -> float:
    return (1.0 + float(annual_rate)) ** (1.0 / 12.0) - 1.0

def _npv_monthly(cash: np.ndarray, r_month: float) -> float:
    t = np.arange(cash.shape[0], dtype=float)
    disc = (1.0 + r_month) ** t
    return float(np.sum(cash / disc))

def _irr_monthly(cash: np.ndarray, guess: float = 0.01, max_iter: int = 100, tol: float = 1e-7) -> float:
    """
    Simple Newton-Raphson IRR on monthly cash flow.
    Returns monthly IRR; caller can annualize.
    """
    r = guess
    t = np.arange(cash.shape[0], dtype=float)

    def f(rate):
        return np.sum(cash / (1.0 + rate) ** t)

    def fp(rate):
        return np.sum(-t * cash / (1.0 + rate) ** (t + 1.0))

    for _ in range(max_iter):
        val = f(r)
        der = fp(r)
        if abs(der) < 1e-12:
            break
        new_r = r - val / der
        if np.isnan(new_r) or np.isinf(new_r):
            break
        if abs(new_r - r) < tol:
            r = new_r
            break
        r = new_r
    return float(r)

def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim to common (n_iter, T)."""
    n1, T1 = a.shape
    n2, T2 = b.shape
    T = min(T1, T2)
    n = min(n1, n2)
    return a[:n, :T], b[:n, :T]

# ----------------------------- core -----------------------------

def build_monthly_vectors(cfg: Dict, horizon_months: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      price_paths: (N,T) IRR/MWh
      energy_paths: (N,T) MWh
      opex_monthly: (T,) IRR per month (deterministic inflation of OPEX)
    """
    n_iter = int(cfg.get("monte_carlo", {}).get("iterations", 1000))
    seed = int(cfg.get("monte_carlo", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed)

    # --- PRICE paths (build proper dataclass config) ---
    price_cfg = _build_price_config(cfg["price"])  # <-- pass the whole YAML, not just cfg["price"]
    price_paths, _, _ = sample_price_path_monthly(
        n_iter=n_iter, months_h=horizon_months, rng=rng, cfg=price_cfg
    )

    # --- ENERGY paths ---
    energy_paths, _, _, _ = sample_production_paths_monthly(
        n_iter=n_iter, months_h=horizon_months, rng=rng, cfg_yaml=cfg
    )

    # shape align
    price_paths, energy_paths = _ensure_same_shape(price_paths, energy_paths)

    # --- OPEX deterministic vector ---
    proj = cfg.get("project", {})
    costs = cfg.get("costs", {})
    cap_mw = float(cfg.get("plant", {}).get("capacity_mw", 20.0))
    irr_per_usd = float(proj.get("irr_per_usd", 1.0))
    opex_usd_kw_yr = float(costs.get("opex_usd_per_kw_yr", 40.0))
    infl_opex_annual = float(costs.get("inflation_opex", 0.20))

    base_opex_irr_year = cap_mw * 1000.0 * opex_usd_kw_yr * irr_per_usd
    r_m = _monthly_rate(infl_opex_annual)
    t = np.arange(price_paths.shape[1], dtype=float)
    opex_monthly = (base_opex_irr_year / 12.0) * (1.0 + r_m) ** t  # (T,)

    return price_paths, energy_paths, opex_monthly

def build_cashflows(cfg: Dict, price_paths: np.ndarray, energy_paths: np.ndarray, opex_monthly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cash_paths: (N,T) monthly cash flows in IRR (CAPEX at t=0 negative)
      revenue_paths: (N,T)
    """
    proj = cfg.get("project", {})
    costs = cfg.get("costs", {})
    cap_mw = float(cfg.get("plant", {}).get("capacity_mw", 20.0))
    irr_per_usd = float(proj.get("irr_per_usd", 1.0))

    capex_usd_kw = float(costs.get("capex_usd_per_kw", 1000.0))
    capex_irr = cap_mw * 1000.0 * capex_usd_kw * irr_per_usd  # paid at t=0

    N, T = price_paths.shape

    # Revenue: MWh * IRR/MWh = IRR
    revenue_paths = energy_paths * price_paths  # (N,T)

    # OPEX broadcast
    opex_b = np.broadcast_to(opex_monthly.reshape(1, T), (N, T))

    # Cash flow (without capex)
    cash_paths = revenue_paths - opex_b

    # Insert CAPEX at t=0
    cash_paths[:, 0] = cash_paths[:, 0] - capex_irr

    return cash_paths, revenue_paths

def evaluate_paths(cfg: Dict, cash_paths: np.ndarray) -> Dict[str, float]:
    """
    Compute NPV & IRR distributions; return summary statistics.
    """
    ann_disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    r_m = _monthly_rate(ann_disc)

    N, T = cash_paths.shape
    npvs = np.zeros(N, dtype=float)
    irr_m = np.zeros(N, dtype=float)

    for i in range(N):
        cf = cash_paths[i]
        npvs[i] = _npv_monthly(cf, r_m)
        try:
            irr_m[i] = _irr_monthly(cf, guess=0.01)
        except Exception:
            irr_m[i] = np.nan

    irr_annual = (1.0 + irr_m) ** 12.0 - 1.0

    def pct(a, q):
        return float(np.nanpercentile(a, q))

    out = {
        "NPV_P5": pct(npvs, 5),
        "NPV_P50": pct(npvs, 50),
        "NPV_P95": pct(npvs, 95),
        "IRR_P5": pct(irr_annual, 5),
        "IRR_P50": pct(irr_annual, 50),
        "IRR_P95": pct(irr_annual, 95),
        "Prob_NPV_Positive": float(np.mean(npvs > 0.0)),
        "Mean_NPV": float(np.nanmean(npvs)),
        "Mean_IRR_Annual": float(np.nanmean(irr_annual)),
    }
    return out

# ----------------------------- main -----------------------------

if __name__ == "__main__":
    CODEBASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(CODEBASE, "config.yaml")
    cfg = _load_yaml_config(CONFIG_PATH)

    # Horizon: 1404/01 → 1414/12 (consistent with other modules)
    start_jy, start_jm = 1404, 1
    end_jy, end_jm = 1414, 12
    horizon_months = (end_jy - start_jy) * 12 + (end_jm - start_jm) + 1
    x_dates = _jalali_month_range(start_jy, start_jm, end_jy, end_jm)

    # Build inputs
    price_paths, energy_paths, opex_monthly = build_monthly_vectors(cfg, horizon_months)

    # Cash flows
    cash_paths, revenue_paths = build_cashflows(cfg, price_paths, energy_paths, opex_monthly)

    # Evaluate
    summary = evaluate_paths(cfg, cash_paths)

    # ---------------- outputs ----------------
    # Save summary CSV
    df_sum = pd.DataFrame([summary])
    df_sum.to_csv(os.path.join(CODEBASE, "economic_summary.csv"), index=False)

    # Save sample time-series (P50) of revenue, opex, cash
    qs = [5, 50, 95]
    rev_q = np.percentile(revenue_paths, qs, axis=0)  # (3,T)
    cash_q = np.percentile(cash_paths, qs, axis=0)
    out_ts = pd.DataFrame({
        "date": pd.to_datetime(x_dates),
        "revenue_P50": rev_q[1],
        "revenue_P5": rev_q[0],
        "revenue_P95": rev_q[2],
        "cash_P50": cash_q[1],
        "cash_P5": cash_q[0],
        "cash_P95": cash_q[2],
        "opex_monthly": opex_monthly
    })
    out_ts.to_csv(os.path.join(CODEBASE, "economic_timeseries_pXX.csv"), index=False)

    # Plots
    # 1) Revenue fan
    plt.figure(figsize=(12, 4))
    plt.plot(x_dates, rev_q[1], label="Revenue P50")
    plt.fill_between(x_dates, rev_q[0], rev_q[2], alpha=0.2, label="Revenue P5–P95")
    plt.title("Monthly Revenue (IRR) — P50 & P5–P95")
    plt.xlabel("Month (Gregorian)")
    plt.ylabel("Revenue (IRR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CODEBASE, "econ_revenue_fan.png"), dpi=150)

    # 2) Cash flow fan
    plt.figure(figsize=(12, 4))
    plt.plot(x_dates, cash_q[1], label="Cash P50")
    plt.fill_between(x_dates, cash_q[0], cash_q[2], alpha=0.2, label="Cash P5–P95")
    plt.title("Monthly Cash Flow (IRR) — P50 & P5–P95")
    plt.xlabel("Month (Gregorian)")
    plt.ylabel("Cash Flow (IRR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CODEBASE, "econ_cash_fan.png"), dpi=150)

    # 3) NPV histogram
    ann_disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    r_m = _monthly_rate(ann_disc)
    N = cash_paths.shape[0]
    npvs = np.array([_npv_monthly(cash_paths[i], r_m) for i in range(N)])
    plt.figure(figsize=(10, 4))
    plt.hist(npvs, bins=60)
    plt.title("NPV Distribution (IRR)")
    plt.xlabel("NPV (IRR)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(CODEBASE, "econ_npv_hist.png"), dpi=150)

    # 4) IRR histogram (annualized)
    irr_m = np.array([_irr_monthly(cash_paths[i], guess=0.01) for i in range(N)])
    irr_a = (1.0 + irr_m) ** 12.0 - 1.0
    irr_a = irr_a[~np.isnan(irr_a)]
    if irr_a.size > 0:
        plt.figure(figsize=(10, 4))
        plt.hist(irr_a, bins=60)
        plt.title("IRR Distribution (Annual)")
        plt.xlabel("IRR (annual)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(CODEBASE, "econ_irr_hist.png"), dpi=150)

    # Console summary
    print("\n=== ECONOMIC SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:,.2f}")
    print("\nSaved: economic_summary.csv, economic_timeseries_pXX.csv, econ_*.png")
