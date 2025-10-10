
# multi_city.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json

import cashflow as cf

def _load_cfg(p: str="config.yaml") -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def list_cities(weibull_csv: str) -> list[str]:
    df = pd.read_csv(weibull_csv)
    cols = [c for c in df.columns if c.lower() in ("city","City","city_name")]
    if cols:
        name_col = cols[0]
    else:
        name_col = df.columns[0]
    cities = sorted(pd.unique(df[name_col].astype(str).str.strip()))
    return [c for c in cities if c and c.lower() != 'nan']

def run_all_cities(cfg_path: str = "config.yaml", optimize_capacity: bool = False) -> str:
    cfg = _load_cfg(cfg_path)
    weib = (cfg.get("wind", {}) or {}).get("weibull_csv_path") or (cfg.get("wind", {}) or {}).get("weibull_csv")
    if not weib:
        raise FileNotFoundError("weibull_csv_path not provided in config.wind")

    cities = list_cities(weib)
    years = int((cfg.get("project", {}) or {}).get("years", 10))
    T = years*12

    out_root = Path("Outputs_multi"); out_root.mkdir(exist_ok=True, parents=True)
    summary_rows=[]

    for city in cities:
        cfg_i = json.loads(json.dumps(cfg))
        cfg_i.setdefault('wind',{})['city']=city

        if optimize_capacity:
            opt=cfg_i.get('optimization',{}) or {}
            cmin=float(opt.get('min_mw',5)); cmax=float(opt.get('max_mw',50)); cstep=float(opt.get('step_mw',5))
            caps=np.arange(cmin,cmax+1e-9,cstep)
            best, table = cf.optimize_capacity(cfg_i, T, caps, objective=str(opt.get('objective','NPV_P50')))
            cfg_i.setdefault('plant',{})['capacity_mw']=float(best)
            table.to_csv(out_root / f"{city}_capacity_scan.csv")

        price_paths, energy_paths, opex_monthly, meta = cf.build_monthly_vectors(cfg_i, T, return_meta=True)
        cash_paths, revenue_paths = cf.build_cashflows(cfg_i, price_paths, energy_paths, opex_monthly)
        summ = cf.evaluate_paths(cfg_i, cash_paths); summ['city']=city
        summary_rows.append(summ)

        city_dir = out_root / city; city_dir.mkdir(exist_ok=True)
        qs=[5,50,95]
        rev_q=np.percentile(revenue_paths,qs,axis=0); cash_q=np.percentile(cash_paths,qs,axis=0)
        future_idx=pd.DatetimeIndex(meta.get('future_idx'))
        ts=pd.DataFrame({'date':pd.to_datetime(future_idx),
                         'rev_p5':rev_q[0],'rev_p50':rev_q[1],'rev_p95':rev_q[2],
                         'cash_p5':cash_q[0],'cash_p50':cash_q[1],'cash_p95':cash_q[2]})
        ts.to_csv(city_dir/"time_series_pcts.csv", index=False)

    summary_df=pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root/"summary_by_city.csv", index=False)
    return str(out_root)
