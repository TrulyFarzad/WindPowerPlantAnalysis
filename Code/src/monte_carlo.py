# monte_carlo.py
# Quick runner to generate vectors, CSVs, and PNGs (USD)

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import cashflow as cf

def _load_cfg(p: str="config.yaml") -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    cfg = _load_cfg("config.yaml")
    years = int((cfg.get("project", {}) or {}).get("years", 10))
    T = years * 12

    price_paths, energy_paths, opex_monthly, meta = cf.build_monthly_vectors(cfg, T, return_meta=True)
    cash_paths, revenue_paths = cf.build_cashflows(cfg, price_paths, energy_paths, opex_monthly)
    summary = cf.evaluate_paths(cfg, cash_paths)

    outdir = Path("Outputs"); outdir.mkdir(parents=True, exist_ok=True)

    # Save summary
    pd.DataFrame([summary]).to_csv(outdir / "economic_summary.csv", index=False)

    # Save P5/P50/P95 time series
    qs = [5, 50, 95]
    rev_q = np.percentile(revenue_paths, qs, axis=0)
    cash_q = np.percentile(cash_paths, qs, axis=0)

    future_idx = pd.DatetimeIndex(meta.get("future_idx"))

    ts = pd.DataFrame({
        "date": pd.to_datetime(future_idx),
        "revenue_P50": rev_q[1], "revenue_P5": rev_q[0], "revenue_P95": rev_q[2],
        "cash_P50": cash_q[1], "cash_P5": cash_q[0], "cash_P95": cash_q[2],
        "opex_monthly_usd": opex_monthly
    })
    ts.to_csv(outdir / "economic_timeseries_pXX.csv", index=False)

    # Plots
    plt.figure(figsize=(12,4))
    plt.plot(future_idx, rev_q[1], label="Revenue P50")
    plt.fill_between(future_idx, rev_q[0], rev_q[2], alpha=0.2, label="Revenue P5–P95")
    plt.legend(); plt.title("Monthly Revenue (USD)"); plt.tight_layout()
    plt.savefig(outdir / "econ_revenue_fan.png", dpi=150); plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(future_idx, cash_q[1], label="Cash P50")
    plt.fill_between(future_idx, cash_q[0], cash_q[2], alpha=0.2, label="Cash P5–P95")
    plt.legend(); plt.title("Monthly Cash Flow (USD)"); plt.tight_layout()
    plt.savefig(outdir / "econ_cash_fan.png", dpi=150); plt.close()

    # NPV / IRR hists
    ann_disc = float((cfg.get("project", {})or{}).get("discount_rate", 0.12))
    r_m = cf._monthly_rate(ann_disc)
    npvs = np.array([cf._npv_monthly(cash_paths[i], r_m) for i in range(cash_paths.shape[0])])
    plt.figure(figsize=(10,4)); plt.hist(npvs, bins=60); plt.title("NPV Distribution (USD)")
    plt.xlabel("USD"); plt.tight_layout(); plt.savefig(outdir / "econ_npv_hist.png", dpi=150); plt.close()

    irr_m = np.array([cf._irr_monthly(cash_paths[i], guess=0.01) for i in range(cash_paths.shape[0])])
    irr_a = (1.0 + irr_m)**12 - 1.0; irr_a = irr_a[~np.isnan(irr_a)]
    if irr_a.size:
        plt.figure(figsize=(10,4)); plt.hist(irr_a, bins=60); plt.title("IRR Distribution (Annual)")
        plt.tight_layout(); plt.savefig(outdir / "econ_irr_hist.png", dpi=150); plt.close()

    print("\n=== ECONOMIC SUMMARY (USD) ===")
    for k, v in summary.items():
        if "IRR" in k: print(f"{k}: {v:.2%}")
        else: print(f"{k}: ${v:,.0f}")
    print(f"\nSaved under {outdir}")

if __name__ == "__main__":
    main()
