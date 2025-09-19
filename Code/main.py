# main.py
# Run end-to-end and emit an HTML report with embedded charts (USD)

from __future__ import annotations
import io, base64, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import src.cashflow as cf  # local
# price_model.py / production_model.py are imported by cashflow internally

def _load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _horizon_months(cfg: dict) -> int:
    proj = cfg.get("project", {}) or {}
    years = int(proj.get("years", 10))
    return years * 12

def _png_dataurl(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

def _fmt_usd(x: float) -> str:
    try:
        return "$" + f"{x:,.0f}"
    except Exception:
        return f"${x}"

def run(cfg_path: str = "config.yaml", out_html: str = "mvp_report.html") -> None:
    # --- resolve paths relative to this file (main.py) ---
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent

    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = (base_dir / cfg_file).resolve()

    out_file = Path(out_html)
    if not out_file.is_absolute():
        out_file = (base_dir / out_file).resolve()

    # --- load config & horizon ---
    cfg = _load_cfg(cfg_file)
    T = _horizon_months(cfg)

    # ---------- Build vectors ----------
    price_paths, energy_paths, opex_monthly, meta = cf.build_monthly_vectors(cfg, T, return_meta=True)
    N, T = price_paths.shape

    # future index (robust conversion)
    future_idx_raw = meta.get("future_idx")
    future_idx = pd.to_datetime(future_idx_raw) if future_idx_raw is not None else pd.date_range(
        pd.Timestamp.utcnow().tz_localize("UTC").to_period("M").to_timestamp(how="start"),
        periods=T, freq="MS", tz="UTC"
    )

    # ---------- Cashflows ----------
    cash_paths, revenue_paths = cf.build_cashflows(cfg, price_paths, energy_paths, opex_monthly)
    summary = cf.evaluate_paths(cfg, cash_paths)

    # ---------- Plots ----------
    qs = [5, 50, 95]
    rev_q = np.percentile(revenue_paths, qs, axis=0)
    cash_q = np.percentile(cash_paths, qs, axis=0)

    # Revenue fan
    fig1 = plt.figure(figsize=(10, 3.6))
    plt.plot(future_idx, rev_q[1], label="Revenue P50")
    plt.fill_between(future_idx, rev_q[0], rev_q[2], alpha=0.2, label="Revenue P5–P95")
    plt.title("Monthly Revenue — P50 & P5–P95")
    plt.xlabel("Month"); plt.ylabel("USD"); plt.legend(); plt.tight_layout()
    img_rev = _png_dataurl(fig1); plt.close(fig1)

    # Cashflow fan
    fig2 = plt.figure(figsize=(10, 3.6))
    plt.plot(future_idx, cash_q[1], label="Cash P50")
    plt.fill_between(future_idx, cash_q[0], cash_q[2], alpha=0.2, label="Cash P5–P95")
    plt.title("Monthly Cash Flow — P50 & P5–P95")
    plt.xlabel("Month"); plt.ylabel("USD"); plt.legend(); plt.tight_layout()
    img_cash = _png_dataurl(fig2); plt.close(fig2)

    # NPV hist (USD)
    ann_disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    r_m = cf._monthly_rate(ann_disc)
    npvs = np.array([cf._npv_monthly(cash_paths[i], r_m) for i in range(N)])
    fig3 = plt.figure(figsize=(8, 3.2))
    plt.hist(npvs, bins=60); plt.title("NPV Distribution (USD)")
    plt.xlabel("USD"); plt.ylabel("Count"); plt.tight_layout()
    img_npv = _png_dataurl(fig3); plt.close(fig3)

    # IRR hist (annual)
    irr_m = np.array([cf._irr_monthly(cash_paths[i], guess=0.01) for i in range(N)])
    irr_a = (1.0 + irr_m)**12 - 1.0
    irr_a = irr_a[~np.isnan(irr_a)]
    img_irr = ""
    if irr_a.size:
        fig4 = plt.figure(figsize=(8, 3.2))
        plt.hist(irr_a, bins=60); plt.title("IRR Distribution (Annual)")
        plt.xlabel("IRR"); plt.ylabel("Count"); plt.tight_layout()
        img_irr = _png_dataurl(fig4); plt.close(fig4)

    # ---------- HTML ----------
    cap_mw = float(cfg.get("plant", {}).get("capacity_mw", 20.0))
    disc = float(cfg.get("project", {}).get("discount_rate", 0.12))
    title = cfg.get("report", {}).get("title", "Wind MVP — Economic Report (USD)")

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:sans-serif;margin:24px}}h1,h2{{margin:.6em 0 .3em}}
.kpi{{display:flex;gap:24px;flex-wrap:wrap}}
.card{{border:1px solid #ddd;border-radius:10px;padding:12px 16px;min-width:220px}}
.muted{{color:#666;font-size:.9em}} img{{border:1px solid #eee;border-radius:8px;margin:10px 0}}
</style></head><body>
<h1>{title}</h1>
<p class="muted">Capacity: {cap_mw:.1f} MW — Discount rate: {disc:.2%} — Scenarios: {N:,} — Horizon: {T} months</p>

<h2>Key KPIs</h2>
<div class="kpi">
  <div class="card"><b>NPV P50:</b><br>{_fmt_usd(summary['NPV_P50'])}</div>
  <div class="card"><b>NPV P10 / P90:</b><br>{_fmt_usd(summary['NPV_P5'])} / {_fmt_usd(summary['NPV_P95'])}</div>
  <div class="card"><b>IRR P50:</b><br>{summary['IRR_P50']*100:.2f}% (annual)</div>
  <div class="card"><b>Prob(NPV>0):</b><br>{summary['Prob_NPV_Positive']*100:.1f}%</div>
</div>

<h2>Charts</h2>
<h3>Revenue</h3>
<img src="{img_rev}" width="100%">
<h3>Cash Flow</h3>
<img src="{img_cash}" width="100%">
<h3>NPV Histogram</h3>
<img src="{img_npv}" width="85%">
{"<h3>IRR Histogram</h3><img src='"+img_irr+"' width='85%'>" if img_irr else ""}

<hr><p class="muted">Meta (price & production):</p>
<pre class="muted">{json.dumps({k:v for k,v in meta.items() if k!='future_idx'}, indent=2)}</pre>
</body></html>"""

    out_file.write_text(html, encoding="utf-8")
    print(f"[ok] Report -> {out_file}")

if __name__ == "__main__":
    run()
