# cashflow.py
# End-to-end economic wiring in USD:
#   - price_model (U.S. dataset) -> price paths [USD/MWh]
#   - production_model -> energy paths [MWh]
#   - OPEX/CAPEX configured in USD
#   - Cashflows/NPV/IRR all in USD

from __future__ import annotations
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# Local modules (same folder)
from . import price_model as pm
from . import production_model as prod

# ---------------- utils ----------------
def _monthly_rate(annual_rate: float) -> float:
    return (1.0 + float(annual_rate)) ** (1.0 / 12.0) - 1.0

def _npv_monthly(cash: np.ndarray, r_month: float) -> float:
    t = np.arange(cash.shape[0], dtype=float)
    disc = (1.0 + r_month) ** t
    return float(np.sum(cash / disc))

def _irr_monthly(cash: np.ndarray, guess: float = 0.01, max_iter: int = 100, tol: float = 1e-7) -> float:
    r = guess
    t = np.arange(cash.shape[0], dtype=float)

    def f(rate):  return np.sum(cash / (1.0 + rate) ** t)
    def fp(rate): return np.sum(-t * cash / (1.0 + rate) ** (t + 1.0))

    for _ in range(max_iter):
        val, der = f(r), fp(r)
        if abs(der) < 1e-12: break
        new_r = r - val / der
        if not np.isfinite(new_r): break
        if abs(new_r - r) < tol: r = new_r; break
        r = new_r
    return float(r)

def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n1, T1 = a.shape; n2, T2 = b.shape
    T = min(T1, T2); n = min(n1, n2)
    return a[:n, :T], b[:n, :T]

# -------------- core --------------
def _price_paths_usd(cfg: Dict, horizon_months: int) -> Tuple[np.ndarray, pd.DatetimeIndex, Dict]:
    """Build monthly USD/MWh price paths from price_model.py (U.S. dataset)."""
    price_cfg = cfg.get("price", {}) or {}
    us_csv     = price_cfg.get("us_csv")
    if not us_csv:
        raise ValueError("cfg.price.us_csv (path to U.S. monthly dataset CSV) is required.")

    state      = price_cfg.get("state", "*")
    sector     = price_cfg.get("sector", "*")
    unit_src   = price_cfg.get("unit_source", "cent_per_kwh")
    weight_col = price_cfg.get("weight_col", "sales")
    tz         = price_cfg.get("tz", "UTC")
    block_len  = int(price_cfg.get("block_len", 12))

    # 1) Monthly nominal series (USD/MWh)
    series_nominal = pm.load_us_dataset(
        csv_path=Path(us_csv), tz=tz, state=state, sector=sector,
        unit_source=unit_src, weight_col=weight_col
    )
    last_price = float(series_nominal.iloc[-1])
    log_returns = pm.compute_log_returns(series_nominal)
    future_idx  = pm.build_future_index(series_nominal.index[-1], horizon_months)

    # 2) Monte Carlo (block bootstrap on log-returns)
    n_iter = int(cfg.get("monte_carlo", {}).get("iterations", 1000))
    seed   = int(cfg.get("monte_carlo", {}).get("random_seed", 42))
    price_paths = pm.simulate_price_paths(
        last_price=last_price,
        log_returns_hist=log_returns,
        n_scenarios=n_iter,
        horizon_m=horizon_months,
        block_len=block_len,
        seed=seed,
    )  # shape (N,T) in USD/MWh

    meta = {
        "price": {
            "us_csv": str(Path(us_csv).absolute()),
            "state": state, "sector": sector, "unit": unit_src,
            "series_start": str(series_nominal.index.min()),
            "series_end": str(series_nominal.index.max()),
            "last_price_usd_mwh": last_price,
            "block_len": block_len, "n_iter": n_iter, "seed": seed,
        }
    }
    return price_paths.astype(float), future_idx, meta

def build_monthly_vectors(cfg: Dict, horizon_months: int, return_meta: bool=False):
    """
    Returns:
      price_paths: (N,T) USD/MWh
      energy_paths: (N,T) MWh
      opex_monthly: (T,) USD per month (deterministic)
      [meta dict if return_meta=True] -> contains future_idx
    """
    # Price paths (USD/MWh) + future index
    price_paths, future_idx, meta_price = _price_paths_usd(cfg, horizon_months)

    # Energy paths (MWh) from production_model (same N, T)
    seed = int(cfg.get("monte_carlo", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed)
    energy_paths, _, meta_prod, _ = prod.sample_production_paths_monthly(
        n_iter=price_paths.shape[0], months_h=horizon_months, rng=rng, cfg_yaml=cfg
    )
    price_paths, energy_paths = _ensure_same_shape(price_paths, energy_paths)

    # OPEX (USD/month): cap_mw * 1000 * opex_usd_per_kw_yr, escalated monthly
    costs = cfg.get("costs", {}) or {}
    cap_mw = float(cfg.get("plant", {}).get("capacity_mw", 20.0))
    opex_usd_kw_yr = float(costs.get("opex_usd_per_kw_yr", 40.0))
    infl_opex_annual = float(costs.get("inflation_opex", 0.20))
    base_opex_year_usd = cap_mw * 1000.0 * opex_usd_kw_yr
    r_m = _monthly_rate(infl_opex_annual)
    t = np.arange(price_paths.shape[1], dtype=float)
    opex_monthly = (base_opex_year_usd / 12.0) * (1.0 + r_m) ** t  # (T,)

    meta_all = {"future_idx": future_idx, "production": meta_prod}
    meta_all.update(meta_price)
    return (price_paths, energy_paths, opex_monthly, meta_all) if return_meta else (price_paths, energy_paths, opex_monthly)

def build_cashflows(cfg: Dict, price_paths: np.ndarray, energy_paths: np.ndarray, opex_monthly: np.ndarray):
    """
    Returns:
      cash_paths: (N,T) USD
      revenue_paths: (N,T) USD
    """
    costs = cfg.get("costs", {}) or {}
    cap_mw = float(cfg.get("plant", {}).get("capacity_mw", 20.0))

    capex_usd_kw = float(costs.get("capex_usd_per_kw", 1000.0))
    capex_usd = cap_mw * 1000.0 * capex_usd_kw  # paid at t=0

    N, T = price_paths.shape

    # Revenue USD = MWh * USD/MWh
    revenue_paths = energy_paths * price_paths  # (N,T)

    # OPEX broadcast
    opex_b = np.broadcast_to(opex_monthly.reshape(1, T), (N, T))

    # Cash flow (with CAPEX at t=0)
    cash_paths = revenue_paths - opex_b
    cash_paths[:, 0] -= capex_usd

    return cash_paths, revenue_paths

def evaluate_paths(cfg: Dict, cash_paths: np.ndarray) -> Dict[str, float]:
    ann_disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    r_m = _monthly_rate(ann_disc)

    N = cash_paths.shape[0]
    npvs = np.array([_npv_monthly(cash_paths[i], r_m) for i in range(N)], dtype=float)
    irr_m = np.array([_irr_monthly(cash_paths[i], guess=0.01) for i in range(N)], dtype=float)
    irr_a = (1.0 + irr_m) ** 12.0 - 1.0

    def pct(a, q): return float(np.nanpercentile(a, q))
    return {
        "NPV_P5": pct(npvs, 5), "NPV_P50": pct(npvs, 50), "NPV_P95": pct(npvs, 95),
        "IRR_P5": pct(irr_a, 5), "IRR_P50": pct(irr_a, 50), "IRR_P95": pct(irr_a, 95),
        "Prob_NPV_Positive": float(np.mean(npvs > 0.0)),
        "Mean_NPV": float(np.nanmean(npvs)), "Mean_IRR_Annual": float(np.nanmean(irr_a)),
    }
