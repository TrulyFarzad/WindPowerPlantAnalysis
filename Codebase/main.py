# -*- coding: utf-8 -*-
"""
main.py — Interactive runner for the Wind MVP

Flow:
  - Load config.yaml
  - Show current assumptions, optionally edit them (interactive)
  - Ask which outputs to generate (plots/HTML only vs. full MC CSVs & Excel bundle)
  - PRICE: sample price paths + plots (fan/heatmap + historical)
  - WIND:  sample energy paths + plots (fan/heatmap + power curve & hist CF)
  - CASHFLOW: build cash & revenue paths
  - ECON SUMMARY: evaluate_paths(cfg, cash_paths)
  - POST: NPV/IRR/Payback CSVs, Tornado, Scenario, CumCash fan (optional)
  - HTML report (embeds PNGs)
"""

from __future__ import annotations
import os, sys, base64
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Allow imports from src/ when running from Codebase/
# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------- Local imports from src -----------------------
from src.price_model import (
    sample_price_path_monthly as price_sample,
    _build_price_config,
)
from src.wind_resource import (
    sample_production_paths_monthly as wind_sample,
    _jalali_month_range,
    _load_scada, _clean_scada, _infer_rated_kw, _fit_power_curve,
    _scada_monthly_cf, ScadaFilters, WindCfg
)
from src.cashflow import (
    _load_yaml_config,
    build_monthly_vectors, build_cashflows,
    _monthly_rate, _npv_monthly, _irr_monthly,
    evaluate_paths,  # signature: evaluate_paths(cfg, cash_paths) -> Dict[str, float]
)
from src.monte_carlo import tornado_analysis, scenario_matrix

# ------------------------- small helpers -------------------------

def _savefig(path: str, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def _embed_img(path: str) -> str:
    if not os.path.exists(path):
        return f'<p><em>Missing figure:</em> {os.path.basename(path)}</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" />'

def _q(paths: np.ndarray, q: float) -> np.ndarray:
    return np.percentile(paths, q, axis=0)

def _qstack(paths: np.ndarray, qs: List[float]) -> np.ndarray:
    return np.vstack([_q(paths, q) for q in qs])

def _align_to_common_T(
    x_dates: List[pd.Timestamp],
    *arrays: np.ndarray
) -> Tuple[List[pd.Timestamp], List[np.ndarray]]:
    Ts = []
    for a in arrays:
        if a is None: continue
        if a.ndim == 2: Ts.append(a.shape[1])
        elif a.ndim == 1: Ts.append(a.shape[0])
        else: Ts.append(a.shape[-1])
    if not Ts:
        return x_dates, list(arrays)
    T_common = min(min(Ts), len(x_dates))
    x_aligned = x_dates[:T_common]
    out = []
    for a in arrays:
        if a is None:
            out.append(a); continue
        if a.ndim == 2: out.append(a[:, :T_common])
        elif a.ndim == 1: out.append(a[:T_common])
        else:
            slicer = [slice(None)]*(a.ndim-1) + [slice(0, T_common)]
            out.append(a[tuple(slicer)])
    return x_aligned, out

def _human(x: float) -> str:
    try:
        if abs(x) >= 1e12: return f"{x/1e12:.2f}e12"
        if abs(x) >= 1e9:  return f"{x/1e9:.2f}e9"
        if abs(x) >= 1e6:  return f"{x/1e6:.2f}e6"
        if abs(x) >= 1e3:  return f"{x/1e3:.2f}k"
        return f"{x:.0f}"
    except Exception:
        return str(x)

# ------------------------- interactive helpers -------------------------

def _yn(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    s = input(prompt + suffix).strip().lower()
    if s == "": return default
    return s in ("y", "yes", "1", "true")

def _ask_float(prompt: str, current: float) -> float:
    s = input(f"{prompt} (current={current}): ").strip()
    if s == "": return current
    try:
        return float(s)
    except Exception:
        print("Invalid number, keeping current.")
        return current

def _ask_int(prompt: str, current: int) -> int:
    s = input(f"{prompt} (current={current}): ").strip()
    if s == "": return current
    try:
        return int(s)
    except Exception:
        print("Invalid integer, keeping current.")
        return current

def _print_config_summary(cfg: Dict[str, Any]):
    proj = cfg.get("project", {})
    plant = cfg.get("plant", {})
    costs = cfg.get("costs", {})
    price = cfg.get("price", {})
    wind  = cfg.get("wind", {})
    mc    = cfg.get("monte_carlo", {})

    print("\n=== Current Assumptions ===")
    print(f"Project: {proj.get('name','(name)')}, Years: {proj.get('years',10)}, Currency: {proj.get('currency','IRR')}")
    print(f"IRR/USD: {proj.get('irr_per_usd',1.0)}, Discount rate (annual): {proj.get('discount_rate',0.12)}")
    print(f"Plant capacity (MW): {plant.get('capacity_mw',20.0)}, Availability: {plant.get('availability',0.97)}, Losses: {plant.get('losses_fraction',0.10)}, Degradation/yr: {plant.get('degradation_per_year',0.007)}")
    print(f"CAPEX (USD/kW): {costs.get('capex_usd_per_kw',1154)}, OPEX (USD/kW-yr): {costs.get('opex_usd_per_kw_yr',43)}, OPEX inflation: {costs.get('inflation_opex',0.20)}")
    print(f"Debt ratio: {costs.get('debt_ratio',1.0)}, Loan rate: {costs.get('loan_rate',0.18)}")
    print(f"Price data path: {price.get('iran_daily_path', '(path)')}, Use VWAP: {price.get('use_vwap', True)}")
    infl = price.get('inflation', {})
    print(f"Price inflation scenario: {infl.get('scenario','baseline')}, baseline annual rate: {infl.get('baseline',{}).get('annual_rate',0.35)}")
    print(f"Wind SCADA path: {wind.get('scada_path','')}, Forecasting path: {wind.get('forecasting_path','')}, power_curve_source: {wind.get('power_curve_source','auto')}")
    print(f"MC iterations: {mc.get('iterations',1000)}, seed: {mc.get('random_seed',42)}")
    print("===========================\n")

def _edit_config_interactively(cfg: Dict[str, Any]) -> Dict[str, Any]:
    proj = cfg.setdefault("project", {})
    plant = cfg.setdefault("plant", {})
    costs = cfg.setdefault("costs", {})
    price = cfg.setdefault("price", {})
    wind  = cfg.setdefault("wind", {})
    mc    = cfg.setdefault("monte_carlo", {})

    # Project
    proj["years"]         = _ask_int("Project years", int(proj.get("years", 10)))
    proj["irr_per_usd"]   = _ask_float("IRR per USD", float(proj.get("irr_per_usd", 1_000_000)))
    proj["discount_rate"] = _ask_float("Discount rate (annual, 0.12=12%)", float(proj.get("discount_rate", 0.12)))

    # Plant
    plant["capacity_mw"]          = _ask_float("Plant capacity (MW)", float(plant.get("capacity_mw", 20.0)))
    plant["availability"]         = _ask_float("Availability (0..1)", float(plant.get("availability", 0.97)))
    plant["losses_fraction"]      = _ask_float("Losses fraction (0..1)", float(plant.get("losses_fraction", 0.10)))
    plant["degradation_per_year"] = _ask_float("Degradation per year (0..1)", float(plant.get("degradation_per_year", 0.007)))

    # Costs
    costs["capex_usd_per_kw"]   = _ask_float("CAPEX (USD/kW)", float(costs.get("capex_usd_per_kw", 1154)))
    costs["opex_usd_per_kw_yr"] = _ask_float("OPEX (USD/kW-yr)", float(costs.get("opex_usd_per_kw_yr", 43)))
    costs["inflation_opex"]     = _ask_float("OPEX inflation (annual)", float(costs.get("inflation_opex", 0.20)))
    costs["debt_ratio"]         = _ask_float("Debt ratio (0..1)", float(costs.get("debt_ratio", 1.0)))
    costs["loan_rate"]          = _ask_float("Loan rate (annual)", float(costs.get("loan_rate", 0.18)))

    # Price (minimal essentials)
    price["use_vwap"] = _yn("Use VWAP from IRENEX data?", bool(price.get("use_vwap", True)))
    infl = price.setdefault("inflation", {})
    base = infl.setdefault("baseline", {})
    base["annual_rate"] = _ask_float("Price inflation baseline (annual)", float(base.get("annual_rate", 0.35)))

    # Wind (only minimal)
    wind["power_curve_source"] = input(f"Power curve source [auto|theoretical|lowess] (current={wind.get('power_curve_source','auto')}): ").strip() or wind.get("power_curve_source","auto")

    # Monte Carlo
    mc["iterations"]  = _ask_int("Monte Carlo iterations", int(mc.get("iterations", 10000)))
    mc["random_seed"] = _ask_int("Random seed", int(mc.get("random_seed", 42)))

    return cfg

def _save_config_yaml(cfg: Dict[str, Any], path: str):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

# ------------------------- main -------------------------

if __name__ == "__main__":
    CODEBASE = ROOT
    CONFIG_PATH = os.path.join(CODEBASE, "config.yaml")
    cfg = _load_yaml_config(CONFIG_PATH)

    # 0) Show current assumptions & optionally edit
    _print_config_summary(cfg)
    if _yn("Do you want to edit these assumptions?", default=False):
        cfg = _edit_config_interactively(cfg)
        _print_config_summary(cfg)
        if _yn("Save updated config.yaml?", default=True):
            _save_config_yaml(cfg, CONFIG_PATH)
            print(f"Saved: {CONFIG_PATH}")

    # 0.1) Ask which outputs to generate
    run_price_plots   = _yn("Run price model & plots?", default=True)
    run_wind_plots    = _yn("Run wind resource model & plots?", default=True)
    run_cashflows     = _yn("Build cashflows (revenue/opex/cash)?", default=True)
    run_post_mc       = _yn("Run post-MC analytics (NPV/IRR/payback, Tornado, Scenario)?", default=True)
    bundle_to_excel   = _yn("Bundle CSV outputs into a single Excel workbook at the end?", default=False)
    gen_html_report   = _yn("Generate the final HTML report?", default=True)

    # Fixed horizon: 1404/01 → 1414/12 (inclusive) => nominal T = 132
    start_jy, start_jm = 1404, 1
    end_jy, end_jm = 1414, 12
    T_nominal = (end_jy - start_jy) * 12 + (end_jm - start_jm) + 1  # 132
    x_dates_full = _jalali_month_range(start_jy, start_jm, end_jy, end_jm)

    # RNG / iterations
    n_iter = int(cfg.get("monte_carlo", {}).get("iterations", 1000))
    seed   = int(cfg.get("monte_carlo", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed)

    # ==================== 1) PRICE ====================
    if run_price_plots:
        price_cfg = _build_price_config(cfg["price"])
        price_paths, price_baseline, monthly_hist = price_sample(
            n_iter=n_iter, months_h=T_nominal, rng=rng, cfg=price_cfg
        )
        x_price, [price_paths, price_baseline] = _align_to_common_T(x_dates_full, price_paths, price_baseline)

        # Fan chart (price)
        qs = [5, 50, 95]
        qmat_price = _qstack(price_paths, qs)
        plt.figure(figsize=(12, 4))
        plt.plot(x_price, price_baseline, label="Baseline (inflation only)", zorder=3)
        plt.plot(x_price, qmat_price[1], label="P50")
        plt.fill_between(x_price, qmat_price[0], qmat_price[2], alpha=0.2, label="P5–P95")
        plt.title("Monthly Energy Price Paths (IRR/MWh) — Baseline + Stochastic Fan")
        plt.xlabel("Month (Gregorian)"); plt.ylabel("Price (IRR/MWh)"); plt.legend()
        _savefig(os.path.join(CODEBASE, "price_sampler_fan_chart.png"))

        # Heatmap (price)
        qs_dense = list(range(5, 100, 5))
        qmat_dense_p = _qstack(price_paths, qs_dense)
        plt.figure(figsize=(12, 5))
        plt.imshow(qmat_dense_p, aspect="auto", origin="lower")
        plt.title("Percentile Heatmap of Simulated Monthly Prices (IRR/MWh)")
        plt.xlabel("Time (months 1404→1414)"); plt.ylabel("Percentile (5→95)")
        xticks = np.linspace(0, qmat_dense_p.shape[1] - 1, 12, dtype=int)
        plt.xticks(xticks, [x_price[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
        plt.yticks(np.arange(len(qs_dense)), [str(q) for q in qs_dense])
        _savefig(os.path.join(CODEBASE, "price_sampler_heatmap.png"))

        # Historical monthly prices
        if monthly_hist is not None and not monthly_hist.empty:
            plt.figure(figsize=(12, 3.6))
            plt.plot(monthly_hist["date"], monthly_hist["price"], label="Historical monthly (IRR/MWh)")
            plt.title("Historical Monthly Energy Prices (IRR/MWh) — IRENEX")
            plt.xlabel("Month (Gregorian)"); plt.ylabel("Price (IRR/MWh)")
            plt.legend()
            _savefig(os.path.join(CODEBASE, "price_hist_monthly.png"))
    else:
        price_paths = price_baseline = None
        x_price = x_dates_full

    # ==================== 2) WIND ====================
    if run_wind_plots:
        wind_paths, wind_baseline, hist_cf, _ = wind_sample(
            n_iter=n_iter, months_h=T_nominal, rng=rng, cfg_yaml=cfg
        )
        x_wind, [wind_paths, wind_baseline] = _align_to_common_T(x_price, wind_paths, wind_baseline)

        qs = [5, 50, 95]
        qmat_wind = _qstack(wind_paths, qs)
        plt.figure(figsize=(12, 4))
        plt.plot(x_wind, wind_baseline, label="Baseline (seasonal × availability × losses × degradation)", zorder=3)
        plt.plot(x_wind, qmat_wind[1], label="P50")
        plt.fill_between(x_wind, qmat_wind[0], qmat_wind[2], alpha=0.2, label="P5–P95")
        plt.title("Monthly Energy Production (MWh) — Baseline + Stochastic Fan")
        plt.xlabel("Month (Gregorian)"); plt.ylabel("Energy (MWh)"); plt.legend()
        _savefig(os.path.join(CODEBASE, "wind_sampler_fan_chart.png"))

        qs_dense = list(range(5, 100, 5))
        qmat_dense_w = _qstack(wind_paths, qs_dense)
        plt.figure(figsize=(12, 5))
        plt.imshow(qmat_dense_w, aspect="auto", origin="lower")
        plt.title("Percentile Heatmap of Simulated Monthly Energy (MWh)")
        plt.xlabel("Time (months 1404→1414)"); plt.ylabel("Percentile (5→95)")
        xticks = np.linspace(0, qmat_dense_w.shape[1] - 1, 12, dtype=int)
        plt.xticks(xticks, [x_wind[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
        plt.yticks(np.arange(len(qs_dense)), [str(q) for q in qs_dense])
        _savefig(os.path.join(CODEBASE, "wind_sampler_heatmap.png"))

        # Optional: power curve & historical CF from SCADA
        try:
            scada = _clean_scada(_load_scada(cfg.get("wind", {}).get("scada_path", "")), ScadaFilters())
            rated_kw = _infer_rated_kw(scada, WindCfg(None, None))
            v_grid, p_grid = _fit_power_curve(scada, rated_kw, source=str(cfg.get("wind", {}).get("power_curve_source","auto")))
            plt.figure(figsize=(8, 4))
            plt.plot(v_grid, p_grid, label="Power curve (LOWESS / theoretical / IEC fallback)")
            plt.title("Empirical Power Curve (kW vs m/s)")
            plt.xlabel("Wind speed (m/s)"); plt.ylabel("Power (kW)"); plt.legend()
            _savefig(os.path.join(CODEBASE, "wind_power_curve.png"))

            hist = _scada_monthly_cf(scada, rated_kw)
            if not hist.empty:
                plt.figure(figsize=(12, 3.2))
                plt.plot(hist["date"], hist["cf"], label="Historical CF (monthly)")
                plt.title("Historical Monthly Capacity Factor (SCADA)")
                plt.xlabel("Month (Gregorian)"); plt.ylabel("Capacity Factor"); plt.ylim(0, 1.0)
                plt.legend()
                _savefig(os.path.join(CODEBASE, "wind_hist_cf.png"))
        except Exception:
            pass
    else:
        wind_paths = wind_baseline = None
        x_wind = x_price

    # Align both modules to a common timeline for cashflow
    x_common = x_wind
    T_common = len(x_common)

    # ==================== 3) CASHFLOWS ====================
    if run_cashflows:
        price_paths_cf, energy_paths_cf, opex_monthly = build_monthly_vectors(cfg, T_common)
        cash_paths, revenue_paths = build_cashflows(cfg, price_paths_cf, energy_paths_cf, opex_monthly)
    else:
        raise SystemExit("Cashflows are required for economic outputs; please enable 'Build cashflows'.")

    # ==================== 4) ECON SUMMARY ====================
    econ_summary: Dict[str, float] = evaluate_paths(cfg, cash_paths)  # your signature
    # (If evaluate_paths writes CSVs internally, they will be created here.)

    # ==================== 5) POST (optional) ====================
    ann_disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    r_m = _monthly_rate(ann_disc)
    N, T = cash_paths.shape
    t = np.arange(T, dtype=float)
    disc = (1.0 + r_m) ** t

    npvs = np.array([_npv_monthly(cash_paths[i], r_m) for i in range(N)], dtype=float)
    irr_m = np.array([_irr_monthly(cash_paths[i], guess=0.01) for i in range(N)], dtype=float)
    irr_a = (1.0 + irr_m) ** 12.0 - 1.0

    # Always helpful CSVs
    pd.DataFrame({"NPV_IRR": npvs}).to_csv(os.path.join(CODEBASE, "mc_npv_distribution.csv"), index=False)
    pd.DataFrame({"IRR_annual": irr_a}).to_csv(os.path.join(CODEBASE, "mc_irr_distribution.csv"), index=False)

    # Discounted payback (months)
    dpb = np.full(N, np.nan, dtype=float)
    for i in range(N):
        cf = cash_paths[i] / disc
        csum = np.cumsum(cf)
        pos = np.where(csum > 0.0)[0]
        if pos.size > 0:
            dpb[i] = float(pos[0])
    pd.DataFrame({"payback_months_discounted": dpb}).to_csv(
        os.path.join(CODEBASE, "mc_payback_distribution.csv"), index=False
    )

    if run_post_mc:
        # Tornado & Scenario
        baseline = {
            "price_paths": price_paths_cf,
            "energy_paths": energy_paths_cf,
            "opex_monthly": opex_monthly,
            "cash_paths": cash_paths,
            "npvs": npvs,
            "irr_annual": irr_a,
            "r_m": r_m,
            "x_dates": x_common,
        }
        tor = tornado_analysis(cfg, baseline)
        tor.to_csv(os.path.join(CODEBASE, "mc_tornado.csv"), index=False)
        # Tornado plot
        plt.figure(figsize=(9, 5))
        y = np.arange(len(tor))
        plt.barh(y, tor["delta_minus"], label="minus")
        plt.barh(y, tor["delta_plus"], label="plus")
        plt.yticks(y, tor["factor"].tolist()); plt.axvline(0.0, color="k")
        plt.title("Tornado: Δ NPV P50 by parameter shocks")
        plt.xlabel("Δ NPV P50 (IRR)"); plt.ylabel("Parameter"); plt.legend()
        _savefig(os.path.join(CODEBASE, "mc_tornado.png"))

        scen_df = scenario_matrix(cfg, baseline,
                                  price_mults=(0.8, 0.9, 1.0, 1.1, 1.2),
                                  cf_mults=(0.9, 1.0, 1.1))
        scen_df.to_csv(os.path.join(CODEBASE, "mc_scenario_matrix.csv"), index=False)
        piv = scen_df.pivot(index="cf_mult", columns="price_mult", values="NPV_P50")
        plt.figure(figsize=(8, 5))
        plt.imshow(piv.values, aspect="auto", origin="lower")
        plt.title("Scenario Matrix: NPV P50 (IRR) by Price×CF multipliers")
        plt.xlabel("Price multiplier"); plt.ylabel("CF multiplier")
        plt.xticks(np.arange(len(piv.columns)), [str(v) for v in list(piv.columns)])
        plt.yticks(np.arange(len(piv.index)), [str(v) for v in list(piv.index)])
        _savefig(os.path.join(CODEBASE, "mc_scenario_heatmap.png"))

        # Cum discounted cash fan
        cumdisc = np.cumsum(cash_paths / disc.reshape(1, T), axis=1)
        qmat_cum = _qstack(cumdisc, [5, 50, 95])
        plt.figure(figsize=(12, 4))
        plt.plot(x_common, qmat_cum[1], label="Cum. discounted cash P50")
        plt.fill_between(x_common, qmat_cum[0], qmat_cum[2], alpha=0.2, label="P5–P95")
        plt.title("Cumulative Discounted Cash Flow — P50 & P5–P95")
        plt.xlabel("Month (Gregorian)"); plt.ylabel("IRR (cumulative, discounted)")
        plt.legend()
        _savefig(os.path.join(CODEBASE, "mc_cumcash_fan.png"))

        if np.isfinite(dpb).any():
            plt.figure(figsize=(9, 4))
            plt.hist(dpb[np.isfinite(dpb)], bins=50)
            plt.title("Discounted Payback Period Distribution")
            plt.xlabel("Months to payback"); plt.ylabel("Frequency")
            _savefig(os.path.join(CODEBASE, "mc_payback_hist.png"))

    # Optional: bundle key CSVs into a single Excel
    if bundle_to_excel:
        xlsx_path = os.path.join(CODEBASE, "mvp_outputs_bundle.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
            # Try to include common CSVs if they exist
            def add_csv(sheet_name, csv_name):
                full = os.path.join(CODEBASE, csv_name)
                if os.path.exists(full):
                    df = pd.read_csv(full)
                    df.to_excel(xw, sheet_name=sheet_name[:31], index=False)
            add_csv("economic_summary", "economic_summary.csv")
            # Many per-path timeseries might exist
            for p in range(1, 6):
                csvn = f"economic_timeseries_p{p:02d}.csv"
                if os.path.exists(os.path.join(CODEBASE, csvn)):
                    add_csv(f"timeseries_p{p:02d}", csvn)
            add_csv("npv_dist", "mc_npv_distribution.csv")
            add_csv("irr_dist", "mc_irr_distribution.csv")
            add_csv("payback_dist", "mc_payback_distribution.csv")
            add_csv("scenario_matrix", "mc_scenario_matrix.csv")
            add_csv("tornado", "mc_tornado.csv")
        print("Excel bundle saved:", xlsx_path)

    # ==================== HTML report (optional) ====================
    if gen_html_report:
        # KPIs from distributions
        npv_p50 = float(np.percentile(npvs, 50))
        npv_p10 = float(np.percentile(npvs, 10))
        npv_p90 = float(np.percentile(npvs, 90))
        irr_p50 = float(np.percentile( (1.0 + np.array([_irr_monthly(cash_paths[i], guess=0.01) for i in range(N)]))**12 - 1.0, 50))
        payback_med = float(np.nanmedian(dpb)) if np.isfinite(dpb).any() else np.nan

        proj_name = cfg.get("project", {}).get("name", "Wind MVP")
        cap_mw    = float(cfg.get("plant", {}).get("capacity_mw", 20.0))
        ann_disc  = float(cfg.get("project", {}).get("discount_rate", 0.12))

        html = f"""<!DOCTYPE html><html lang="fa"><head><meta charset="utf-8">
<title>{proj_name} — MVP Report</title>
<style>body{{font-family:sans-serif;margin:24px}}h1,h2{{margin:.6em 0 .3em}}.kpi{{display:flex;gap:24px;flex-wrap:wrap}}
.card{{border:1px solid #ddd;border-radius:10px;padding:12px 16px;min-width:220px}}.muted{{color:#666;font-size:.9em}}
img{{border:1px solid #eee;border-radius:8px;margin:10px 0}}</style></head><body>
<h1>{proj_name} — گزارش MVP</h1>
<p class="muted">دوره تحلیل: 1404/01 تا 1414/12 — ظرفیت اسمی: {cap_mw:.1f} MW — نرخ تنزیل: {ann_disc:.2%}</p>
<h2>شاخص‌های اقتصادی کلیدی</h2>
<div class="kpi">
  <div class="card"><b>NPV P50:</b><br>{_human(npv_p50)} IRR</div>
  <div class="card"><b>NPV P10 / P90:</b><br>{_human(npv_p10)} / {_human(npv_p90)} IRR</div>
  <div class="card"><b>IRR P50:</b><br>{irr_p50:.2%} سالانه</div>
  <div class="card"><b>Median Payback (disc.):</b><br>{'∞' if not np.isfinite(payback_med) else f'{int(payback_med)} ماه'}</div>
</div>
<h2>مدل قیمت برق</h2>
{_embed_img(os.path.join(CODEBASE, "price_hist_monthly.png"))}
{_embed_img(os.path.join(CODEBASE, "price_sampler_fan_chart.png"))}
{_embed_img(os.path.join(CODEBASE, "price_sampler_heatmap.png"))}
<h2>منابع باد و تولید انرژی</h2>
{_embed_img(os.path.join(CODEBASE, "wind_power_curve.png"))}
{_embed_img(os.path.join(CODEBASE, "wind_hist_cf.png"))}
{_embed_img(os.path.join(CODEBASE, "wind_sampler_fan_chart.png"))}
{_embed_img(os.path.join(CODEBASE, "wind_sampler_heatmap.png"))}
<h2>آنالیز Monte Carlo</h2>
{_embed_img(os.path.join(CODEBASE, "mc_cumcash_fan.png"))}
{_embed_img(os.path.join(CODEBASE, "mc_payback_hist.png"))}
{_embed_img(os.path.join(CODEBASE, "mc_tornado.png"))}
{_embed_img(os.path.join(CODEBASE, "mc_scenario_heatmap.png"))}
<p class="muted">CSVها: economic_summary.csv (در صورت تولید توسط cashflow)، economic_timeseries_pXX.csv،
mc_npv_distribution.csv، mc_irr_distribution.csv، mc_payback_distribution.csv، mc_scenario_matrix.csv و ...</p>
</body></html>"""
        report_path = os.path.join(CODEBASE, "mvp_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print("Report saved:", report_path)

    print("\nDone.")
