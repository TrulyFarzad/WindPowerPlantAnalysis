# Code/main.py — Jinja2-based reporting with P50 line/bars + P5–P95 band
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime

import src.cashflow as cf  # local economic engine
from jinja2 import Environment, FileSystemLoader, select_autoescape
from src.report_utils import save_fig_dual, timestamped_assets_dir

# ---------- styles (light/dark/auto) ----------
def _style_block(theme: str = "auto") -> str:
    base = """
:root {--bg:#ffffff;--fg:#0f172a;--muted:#64748b;--card:#f8fafc;--border:#e2e8f0;--accent:#2563eb;}
html,body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;margin:24px;line-height:1.5}
h1,h2,h3{margin:.6em 0 .3em}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.kpi{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0}
.card{border:1px solid var(--border);border-radius:12px;padding:12px 16px;min-width:220px;background:var(--card)}
.muted{color:var(--muted);font-size:.9em}
img{border:1px solid var(--border);border-radius:10px;margin:10px 0;max-width:100%;height:auto}
table.kpitable{border-collapse:collapse;width:100%;margin:10px 0}
table.kpitable td,table.kpitable th{border:1px solid var(--border);padding:8px;text-align:left}
.section{margin-top:28px}
small.code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;color:var(--muted)}
"""
    dark = """
:root {--bg:#0b0d10;--fg:#e8eaed;--muted:#a1a1aa;--card:#14181d;--border:#30363d;--accent:#60a5fa;}
"""
    if theme == "dark":
        return dark + base
    if theme == "light":
        return base
    return base + "@media (prefers-color-scheme: dark){" + dark + "}"

# ---------- helpers ----------
def fmt_money(x):
    try:
        return "$" + f"{float(x):,.0f}"
    except Exception:
        return str(x)

def fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return str(x)

def compute_percentiles(npv_samples=None, irr_samples=None):
    pct = {}
    if npv_samples is not None and len(npv_samples) > 0:
        p = np.percentile(npv_samples, [5, 50, 95])
        pct["npv"] = {"p5": fmt_money(p[0]), "p50": fmt_money(p[1]), "p95": fmt_money(p[2])}
    if irr_samples is not None and len(irr_samples) > 0:
        p = np.percentile(irr_samples, [5, 50, 95])
        pct["irr"] = {"p5": fmt_pct(p[0]), "p50": fmt_pct(p[1]), "p95": fmt_pct(p[2])}
    return pct or None

def _plot_band_line(x, p50, p5, p95, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    ax.fill_between(x, p5, p95, alpha=0.15, lw=0, label="P5–P95")
    ax.plot(x, p50, lw=2.2, label="P50")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    return fig

def _plot_cashflow_band_bars(
    x, p50, p5, p95,
    title, xlabel, ylabel,
    scale_mode: str = "robust",   # "robust" | "fixed"
    fixed_ylim_abs: float = None, # اگر fixed انتخاب شد، مثلا 1e6 بده
    expand: float = 1.2           # ضریب بزرگ‌نمایی روی مقیاس روباست
):
    """
    Bar chart for P50 with asymmetric error bars (P5–P95).
    y-axis is centered at 0. For scaling:
      - 'robust': compute symmetric limits from NON-CAPEX months (exclude month 0)
      - 'fixed' : use fixed symmetrical limits ±fixed_ylim_abs
    If CAPEX is outside the limits, annotate it with an arrow & value.
    """
    fig, ax = plt.subplots(figsize=(10.5, 3.6))

    # میله‌های P50
    ax.bar(x, p50, width=0.8, align="center", label="P50")

    # خطاهای نامتقارن برای باند P5–P95
    lower = np.maximum(0.0, p50 - p5)
    upper = np.maximum(0.0, p95 - p50)
    ax.errorbar(x, p50, yerr=[lower, upper], fmt="none", alpha=0.5, label="P5–P95")

    # --- تعیین مقیاس ---
    if scale_mode == "fixed" and fixed_ylim_abs:
        M = float(fixed_ylim_abs)
    else:
        # مقیاس «روباست»: فقط از ماه‌های عملیاتی (غیر از CAPEX اول) استفاده کن
        if len(p50) > 1:
            arr = np.vstack([np.abs(p5[1:]), np.abs(p50[1:]), np.abs(p95[1:])])
        else:
            arr = np.vstack([np.abs(p5), np.abs(p50), np.abs(p95)])
        # از صدک ۹۹ برای مقاومت در برابر نقاط پرت استفاده می‌کنیم
        M = float(np.nanpercentile(arr, 99))
        if not np.isfinite(M) or M == 0:
            M = float(np.nanmax(arr) or 1.0)
        M *= float(expand)

    # محور y متقارن حول صفر
    ax.set_ylim(-M, M)
    ax.axhline(0.0, lw=1.0, alpha=0.5)

    # اگر CAPEX (اولی) خارج از محدوده است، با فلش و لیبل نشانش بده
    try:
        capex_y = p50[0]
        if capex_y < -M or capex_y > M:
            y_edge = -M if capex_y < 0 else M
            ax.annotate(
                fmt_money(capex_y) + " (clipped)",
                xy=(x[0], y_edge),
                xytext=(x[0], y_edge * 0.85 if capex_y < 0 else y_edge * 0.85),
                textcoords="data",
                ha="center", va="top" if capex_y < 0 else "bottom",
                arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.7),
                fontsize=8, rotation=90,
            )
    except Exception:
        pass

    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25); ax.legend()
    fig.tight_layout()
    return fig


# ---------- main entry ----------
def run(cfg: dict, out_html: str = "mvp_report.html") -> None:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict built by the Web UI")

    project = cfg.get("project", {}) or {}
    years = float(project.get("years", project.get("horizon_years", 20.0)))
    horizon_months = int(round(years * 12))

    price_paths, energy_paths, opex_monthly, meta = cf.build_monthly_vectors(
        cfg, horizon_months, return_meta=True
    )
    cash_paths, revenue_paths = cf.build_cashflows(cfg, price_paths, energy_paths, opex_monthly)
    kpis = cf.evaluate_paths(cfg, cash_paths)

    # percentiles along simulation axis (axis=0 if shape is (n_sims, n_months))
    def pct_band(arr):
        p = np.percentile(arr, [5, 50, 95], axis=0)
        return p[0], p[1], p[2]  # p5, p50, p95

    months = np.arange(1, horizon_months + 1)

    # Production — Monthly (line band)
    p5_m, p50_m, p95_m = pct_band(energy_paths)
    fig_prod_m = _plot_band_line(months, p50_m, p5_m, p95_m, "Production — Monthly P50 & P5–P95", "Month", "MWh")

    # Production — Annual (sum each 12 months per sim)
    n_sims = energy_paths.shape[0]
    years_count = horizon_months // 12
    energy_paths_trim = energy_paths[:, :years_count*12]
    energy_annual = energy_paths_trim.reshape(n_sims, years_count, 12).sum(axis=2)
    years_index = np.arange(1, years_count + 1)
    p5_y, p50_y, p95_y = pct_band(energy_annual)
    fig_prod_y = _plot_band_line(years_index, p50_y, p5_y, p95_y, "Production — Annual P50 & P5–P95", "Year", "MWh")

    # Price — Monthly (line band)
    p5_p, p50_p, p95_p = pct_band(price_paths)
    fig_price = _plot_band_line(months, p50_p, p5_p, p95_p, "Price — P50 & P5–P95", "Month", "USD/MWh")

    # Revenue — Monthly (line band, طبق خواسته بعد از دو نمودار بالا)
    p5_r, p50_r, p95_r = pct_band(revenue_paths)
    fig_rev = _plot_band_line(months, p50_r, p5_r, p95_r, "Monthly Revenue — P50 & P5–P95", "Month", "USD")

    # Cashflow — Monthly (BAR + band via errorbar, y=0 centered)
    p5_c, p50_c, p95_c = pct_band(cash_paths)
    fig_cashbar = _plot_cashflow_band_bars(months, p50_c, p5_c, p95_c,
                                           "Monthly Cashflow — P50 (bars) & P5–P95",
                                           "Month", "USD")

    # Save charts (PNG + Data URI)
    assets_dir = timestamped_assets_dir(base_dir="report_assets")
    charts_ctx = {}
    du, pth = save_fig_dual(fig_prod_y, assets_dir, "production_annual"); plt.close(fig_prod_y)
    charts_ctx["production_annual"] = {"data_uri": du, "png_path": pth}
    du, pth = save_fig_dual(fig_prod_m, assets_dir, "production_monthly"); plt.close(fig_prod_m)
    charts_ctx["production_monthly"] = {"data_uri": du, "png_path": pth}
    du, pth = save_fig_dual(fig_price, assets_dir, "price"); plt.close(fig_price)
    charts_ctx["price"] = {"data_uri": du, "png_path": pth}
    du, pth = save_fig_dual(fig_rev, assets_dir, "revenue_monthly"); plt.close(fig_rev)
    charts_ctx["revenue_monthly"] = {"data_uri": du, "png_path": pth}
    du, pth = save_fig_dual(fig_cashbar, assets_dir, "cashflow_monthly"); plt.close(fig_cashbar)
    charts_ctx["cashflow_monthly"] = {"data_uri": du, "png_path": pth}

    # NPV/IRR distributions (histograms)
    from src.cashflow import _npv_monthly, _monthly_rate
    npvs = [_npv_monthly(c, _monthly_rate(float(project.get("discount_rate", 0.12)))) for c in cash_paths]
    fig_npv = plt.figure(figsize=(7.5, 3.3))
    plt.hist(npvs, bins=40); plt.title("NPV Distribution (USD)"); plt.grid(True, alpha=0.3)
    du, pth = save_fig_dual(fig_npv, assets_dir, "npv_hist"); plt.close(fig_npv)
    charts_ctx["npv_hist"] = {"data_uri": du, "png_path": pth}

    irr_samples = None
    try:
        from src.cashflow import _irr_monthly
        irr_m = [_irr_monthly(c, guess=0.01) for c in cash_paths]
        irr_a = (1.0 + np.array(irr_m, float)) ** 12.0 - 1.0
        irr_samples = irr_a[np.isfinite(irr_a)]
        fig_irr = plt.figure(figsize=(7.5, 3.3))
        plt.hist(irr_samples, bins=40); plt.title("IRR Distribution (Annualized)"); plt.grid(True, alpha=0.3)
        du, pth = save_fig_dual(fig_irr, assets_dir, "irr_hist"); plt.close(fig_irr)
        charts_ctx["irr_hist"] = {"data_uri": du, "png_path": pth}
    except Exception:
        pass

    # Percentiles table for npv/irr
    percentiles = compute_percentiles(npv_samples=npvs, irr_samples=irr_samples)

    # KPI cards
    kpi_cards = []
    for key in ["NPV_P50", "NPV_P5", "NPV_P95", "IRR_P50", "Prob_NPV_Positive"]:
        if key in kpis:
            if key.startswith("NPV"):
                val = fmt_money(kpis[key])
            elif key.startswith("IRR") or "Prob" in key:
                val = fmt_pct(kpis[key])
            else:
                val = kpis[key]
            kpi_cards.append({"label": key.replace("_"," "), "value": val, "note": None})

    # Meta
    safe_meta = {k: v for k, v in (meta or {}).items() if k != "future_idx"}
    meta_pretty = json.dumps(safe_meta, ensure_ascii=False, indent=2)

    # Render Jinja2
    theme = (cfg.get("report") or {}).get("theme", "auto")
    style_css = _style_block(theme)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("reporting.html.j2")

    context = {
        "title": "Wind MVP — Economic Report",
        "header": "گزارش اقتصادی نیروگاه بادی",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "currency": "USD",
        "style_css": style_css,
        "kpi_cards": kpi_cards,
        "charts": charts_ctx,
        "percentiles": percentiles,
        "kpis_full": {
            k: (fmt_money(v) if any(x in k for x in ["NPV","Revenue","Cash","CapEx"])
               else (fmt_pct(v) if "IRR" in k else v))
            for k, v in (kpis or {}).items()
        },
        "meta_pretty": meta_pretty,
    }

    out_path = (Path(__file__).parent / out_html) if not Path(out_html).is_absolute() else Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = tpl.render(**context)
    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] Report -> {out_path}")
    print(f"[ok] Assets -> {assets_dir}")

if __name__ == "__main__":
    raise SystemExit("Launch via Web UI (backend_adapter → main.run(cfg)).")
