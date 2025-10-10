
# monte_carlo.py — library runner (no config file reads)
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import cashflow as cf  # relative import when src/ is on sys.path

def run(cfg: dict, outdir: str = "Outputs"):
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict built by the Web UI")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # horizon
    years = float((cfg.get("project") or {}).get("years", 20.0))
    T = int(round(years * 12))

    price_paths, energy_paths, opex_monthly, meta = cf.build_monthly_vectors(cfg, T, return_meta=True)
    cash_paths, _ = cf.build_cashflows(cfg, price_paths, energy_paths, opex_monthly)
    summary = cf.evaluate_paths(cfg, cash_paths)

    # plots
    npvs = []
    from cashflow import _npv_monthly, _monthly_rate, _irr_monthly
    r_m = _monthly_rate(float((cfg.get("project") or {}).get("discount_rate", 0.12)))
    for i in range(cash_paths.shape[0]):
        npvs.append(_npv_monthly(cash_paths[i], r_m))
    npvs = np.array(npvs, float)
    plt.figure(figsize=(10,4)); plt.hist(npvs, bins=60); plt.title("NPV Distribution (USD)")
    plt.tight_layout(); plt.savefig(outdir / "econ_npv_hist.png", dpi=150); plt.close()

    irr_m = np.array([_irr_monthly(cash_paths[i], guess=0.01) for i in range(cash_paths.shape[0])])
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
    raise SystemExit("Use Web UI to build cfg and call monte_carlo.run(cfg, outdir).")
