# -*- coding: utf-8 -*-
"""
price_model.py — بورس انرژی (ایران)، مدل ماهانه قیمت با تورم و نوسان تفکیک‌شده.

بهبودها:
- Excel/CSV ingestion (Persian-friendly), VWAP/AVG روزانه -> ماهانه
- Trend: log-linear + ماه-از-سال -> residuals (ایستا)
- Volatility:
    * bootstrap با سطل‌بندی: month_of_year | season3 | none
    * student_t (df=nu) به‌جای بوت‌استرپ در صورت نیاز
    * کالیبراسیون scale به σ هدف ماهانه (روی log-price)
- Inflation scenarios: baseline | piecewise | custom_path + pass_through
- __main__: نمودار ماهانه خام + Fan + Heatmap (1404→1414، 1000 مسیر)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import jdatetime
    HAS_JDATE = True
except Exception:
    HAS_JDATE = False

# ========================= Config =========================

@dataclass
class VolCalib:
    target_monthly_sigma: Optional[float] = None  # σ هدف روی log(price)
    hard_min: float = 0.5
    hard_max: float = 2.0

@dataclass
class VolStudentT:
    nu: float = 5.0  # degrees of freedom

@dataclass
class VolatilityConfig:
    sampler: str = "bootstrap"     # bootstrap | student_t
    by_bucket: str = "month_of_year"  # month_of_year | season3 | none
    scale: float = 1.0
    calibrate_sigma: Optional[VolCalib] = None
    student_t: Optional[VolStudentT] = None

@dataclass
class BaselineInflation:
    annual_rate: float = 0.35

@dataclass
class PiecewiseInflationStep:
    months: int
    annual_rate: float

@dataclass
class CustomPathInflation:
    monthly_rates: List[float]

@dataclass
class InflationConfig:
    scenario: str = "baseline"  # baseline | piecewise | custom_path
    pass_through: float = 1.0
    baseline: Optional[BaselineInflation] = None
    piecewise: Optional[List[PiecewiseInflationStep]] = None
    custom_path: Optional[CustomPathInflation] = None

@dataclass
class PriceConfig:
    iran_daily_path: str
    use_vwap: bool
    fallback_price_irr_per_mwh: float
    volatility: VolatilityConfig
    inflation: InflationConfig

# ========================= Helpers =========================

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "00112233445566778899")
def to_ascii_digits(val: str) -> str:
    return val.translate(PERSIAN_DIGITS)

def clean_number(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip()
    if not s: return None
    s = to_ascii_digits(s).replace("٬","").replace(",","").replace(" ","").replace("٫",".")
    s = "".join(ch for ch in s if ch in "0123456789.-+eE")
    if s in ("","+","-",".","e","E"): return None
    try: return float(s)
    except: return None

def jalali_to_gregorian_str(s: str) -> Optional[str]:
    if not HAS_JDATE: return None
    try:
        s2 = to_ascii_digits(s).replace("-", "/").replace(".", "/")
        y,m,d = map(int, s2.split("/")[:3])
        g = jdatetime.date(y,m,d).togregorian()
        return f"{g.year:04d}-{g.month:02d}-{g.day:02d}"
    except: return None

# ========================= Loaders (CSV/Excel) =========================

def _pick_price_column(df: pd.DataFrame, use_vwap: bool) -> str:
    cols = [c.lower() for c in df.columns]
    if use_vwap and "vwap_price_irr_per_mwh" in cols:
        return df.columns[cols.index("vwap_price_irr_per_mwh")]
    if "avg_price_irr_per_mwh" in cols:
        return df.columns[cols.index("avg_price_irr_per_mwh")]
    cand = [c for c in df.columns if re.search(r"price", c, re.I)]
    if cand: return cand[0]
    raise ValueError("CSV: expected 'vwap_price_irr_per_mwh' or 'avg_price_irr_per_mwh'.")

def _coerce_date_cols_csv(df: pd.DataFrame) -> pd.Series:
    lc = {c.lower(): c for c in df.columns}
    if "date" in lc:
        s = pd.to_datetime(df[lc["date"]], errors="coerce")
        if s.notna().any(): return s
    if "date_jalali" in lc and HAS_JDATE:
        g = df[lc["date_jalali"]].apply(lambda x: jalali_to_gregorian_str(str(x)))
        return pd.to_datetime(g, errors="coerce")
    return pd.to_datetime(df.iloc[:,0], errors="coerce")

def _load_from_csv(path: str, use_vwap: bool) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    date_series = _coerce_date_cols_csv(df)
    price_col = _pick_price_column(df, use_vwap)
    price = df[price_col].apply(clean_number)
    out = pd.DataFrame({"date": date_series, "price": price})
    return out.dropna(subset=["date","price"]).sort_values("date").reset_index(drop=True)

_PERSIAN_DATE_KEYS = ["تاریخ","تاريخ","date","tarikh"]
_PERSIAN_PRICE_KEYS= ["قیمت","قيمت","price","میانگین","ميانگين","میانگین قیمت","میانگین موزون","settlement","final","vwap","avg"]
_PERSIAN_VOL_KEYS  = ["حجم","مقدار","quantity","vol","volume","حجم معاملات","ارزش","ارزش معاملات"]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _is_date_col(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s): return True
    if s.dtype == object:
        sample = s.dropna().astype(str).head(20).tolist()
        hits = 0
        for val in sample:
            try: pd.to_datetime(val); hits += 1; continue
            except: pass
            if jalali_to_gregorian_str(val) is not None: hits += 1
        return hits >= max(3, len(sample)//4)
    return False

def _find_col(cols: List[str], keys: List[str]) -> List[str]:
    out=[]
    for c in cols:
        lc=c.lower()
        if any(k in lc for k in keys): out.append(c)
    return out

def _load_from_excel(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    daily_list=[]
    for sh in xls.sheet_names:
        try: df = pd.read_excel(path, sheet_name=sh)
        except: continue
        if df is None or df.empty: continue
        df = _normalize_cols(df)

        date_col=None
        for c in df.columns:
            if _is_date_col(df[c]): date_col=c; break
        if date_col is None:
            cand=_find_col(list(df.columns), _PERSIAN_DATE_KEYS)
            if cand: date_col=cand[0]
        if date_col is None: continue

        sdate = df[date_col]
        if pd.api.types.is_datetime64_any_dtype(sdate):
            d = pd.to_datetime(sdate, errors="coerce")
        else:
            d = pd.to_datetime(sdate, errors="coerce")
            if d.isna().mean()>0.5 and HAS_JDATE:
                d = pd.to_datetime(sdate.apply(lambda x: jalali_to_gregorian_str(str(x))), errors="coerce")

        price_cands = _find_col(list(df.columns), _PERSIAN_PRICE_KEYS)
        vol_cands   = _find_col(list(df.columns), _PERSIAN_VOL_KEYS)

        price_cols_numeric=[]
        for c in price_cands:
            num = df[c].apply(clean_number)
            if pd.Series(num).notna().sum() >= max(5, len(df)*0.1):
                price_cols_numeric.append((c,num))
        if not price_cols_numeric:
            for c in df.columns:
                num = df[c].apply(clean_number)
                if pd.Series(num).notna().sum() >= max(5, len(df)*0.2):
                    price_cols_numeric.append((c,num))
        if not price_cols_numeric: continue

        def _score(name:str)->int:
            n=name.lower(); score=0
            for k in ["vwap","میانگین موزون","settlement","final","price","قیمت","میانگین","avg"]:
                if k in n: score+=1
            return score
        price_cols_numeric.sort(key=lambda t:_score(t[0]), reverse=True)
        price_series = price_cols_numeric[0][1]

        vol_series=None
        for c in vol_cands:
            num=df[c].apply(clean_number)
            if pd.Series(num).notna().sum() >= max(5, len(df)*0.1):
                vol_series=num; break

        day = pd.to_datetime(pd.to_datetime(d).dt.date, errors="coerce")
        tmp = pd.DataFrame({"date":day, "price":price_series})
        if vol_series is not None: tmp["vol"]=vol_series
        tmp = tmp.dropna(subset=["date","price"])
        if tmp.empty: continue

        if "vol" in tmp.columns and tmp["vol"].fillna(0).sum()>0:
            grp = tmp.groupby("date", as_index=False).apply(
                lambda g: pd.Series({
                    "sum_pv": float((g["price"]*g.get("vol",0)).sum()),
                    "sum_v" : float(g.get("vol",0).sum()),
                    "avg"   : float(g["price"].mean())
                })
            ).reset_index(drop=True)
            grp["price"] = np.where(grp["sum_v"]>0, grp["sum_pv"]/grp["sum_v"], grp["avg"])
            daily_list.append(grp[["date","price"]])
        else:
            grp = tmp.groupby("date", as_index=False)["price"].mean()
            daily_list.append(grp)

    if not daily_list: return pd.DataFrame(columns=["date","price"])
    daily = pd.concat(daily_list, ignore_index=True)
    daily = daily.dropna(subset=["date","price"])
    daily = daily.groupby("date", as_index=False)["price"].mean().sort_values("date").reset_index(drop=True)
    return daily

def load_iran_daily(cfg: PriceConfig) -> pd.DataFrame:
    path = cfg.iran_daily_path
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["date","price"])
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv",".txt"]:
        return _load_from_csv(path, cfg.use_vwap)
    return _load_from_excel(path)

# ========================= Monthly & Trend =========================

def aggregate_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty: return df_daily
    x=df_daily.copy()
    x["date"]=pd.to_datetime(x["date"])
    x["year"]=x["date"].dt.year
    x["month"]=x["date"].dt.month
    m = x.groupby(["year","month"], as_index=False)["price"].mean()
    m["date"]=pd.to_datetime(m["year"].astype(str)+"-"+m["month"].astype(str)+"-01")+pd.offsets.MonthEnd(0)
    return m[["date","price"]].sort_values("date").reset_index(drop=True)

def _design_matrix(monthly: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    y = np.log(monthly["price"].values.astype(float))
    n=len(monthly)
    t=np.arange(n).astype(float).reshape(-1,1)
    months = monthly["date"].dt.month.values
    dummies = np.zeros((n,11), dtype=float)  # Dec ref
    for i,mo in enumerate(months):
        if 1<=mo<=11: dummies[i,mo-1]=1.0
    X = np.hstack([np.ones((n,1)), t, dummies])
    return X,y,months

def _ols_solve(X:np.ndarray, y:np.ndarray)->np.ndarray:
    beta,*_ = np.linalg.lstsq(X,y,rcond=None)
    return beta

def fit_trend_residuals(monthly: pd.DataFrame)->Dict[str,np.ndarray]:
    X,y,months=_design_matrix(monthly)
    beta=_ols_solve(X,y)
    fitted=X@beta
    resid=y-fitted
    return {
        "beta":beta,
        "months":months,
        "fitted_log":fitted,
        "residuals":resid,
        "last_price":float(monthly["price"].iloc[-1]),
        "last_month":int(monthly["date"].iloc[-1].month),
        "n_obs":len(monthly),
        "sigma_hist": float(np.std(resid, ddof=1)) if len(resid)>1 else 0.0
    }

# ========================= Inflation index =========================

def _annual_to_monthly_rate(annual: float)->float:
    return (1.0+annual)**(1.0/12.0)-1.0

def build_inflation_index(months:int, infl:InflationConfig)->np.ndarray:
    idx=np.ones(months, dtype=float)
    if infl.scenario=="baseline" and infl.baseline:
        r_m=_annual_to_monthly_rate(float(infl.baseline.annual_rate))
        for t in range(1,months): idx[t]=idx[t-1]*(1.0+r_m)
    elif infl.scenario=="piecewise" and infl.piecewise:
        t=0
        for step in infl.piecewise:
            r_m=_annual_to_monthly_rate(float(step.annual_rate))
            for _ in range(int(step.months)):
                idx[t]= idx[t-1]*(1.0+r_m) if t>0 else 1.0
                t+=1
                if t>=months: break
            if t>=months: break
        while t<months:
            idx[t]=idx[t-1]; t+=1
    elif infl.scenario=="custom_path" and infl.custom_path:
        rates=list(infl.custom_path.monthly_rates)
        for t in range(1,months):
            r = rates[t-1] if t-1<len(rates) else (rates[-1] if rates else 0.0)
            idx[t]=idx[t-1]*(1.0+float(r))
    if infl.pass_through!=1.0:
        idx = idx**float(infl.pass_through)
    return idx

# ========================= Volatility buckets & sampler =========================

def _bucket_key(mo:int, mode:str)->int:
    if mode=="month_of_year": return mo
    if mode=="season3":
        if mo in [12,1,2]: return 0   # سرد
        if mo in [6,7,8]:  return 2   # گرم
        return 1                      # میانه
    return -1  # none

def _residual_buckets(residuals:np.ndarray, months:np.ndarray, mode:str)->Dict[int,np.ndarray]:
    if mode=="none": return {-1: residuals}
    buckets={}
    for i,mo in enumerate(months):
        k=_bucket_key(int(mo), mode)
        buckets.setdefault(k, []).append(residuals[i])
    for k in list(buckets.keys()):
        buckets[k]=np.array(buckets[k], dtype=float)
    return buckets

def _calibrate_scale(hist_sigma:float, cfg:VolatilityConfig)->float:
    cal=cfg.calibrate_sigma
    if not cal or cal.target_monthly_sigma is None or hist_sigma<=0: 
        return float(cfg.scale)
    s = float(cal.target_monthly_sigma) / float(hist_sigma)
    s = max(float(cal.hard_min), min(float(cal.hard_max), s))
    return s

def _draw_residuals(n:int, pool:np.ndarray, rng:np.random.Generator, cfg:VolatilityConfig)->np.ndarray:
    if cfg.sampler=="student_t":
        nu = float((cfg.student_t.nu if cfg.student_t else 5.0))
        # scale residuals to pool's std
        sig = np.std(pool, ddof=1) if pool.size>1 else 0.0
        z = rng.standard_t(df=nu, size=n)
        return z * (sig / np.sqrt(nu/(nu-2))) if nu>2 and sig>0 else z*0.0
    # default: bootstrap
    return rng.choice(pool, size=n, replace=True)

def sample_price_path_monthly(
    n_iter:int, months_h:int, rng:np.random.Generator, cfg:PriceConfig
)->Tuple[np.ndarray,np.ndarray,pd.DataFrame]:
    daily = load_iran_daily(cfg)
    if daily.empty:
        idx=build_inflation_index(months_h, cfg.inflation)
        base=float(cfg.fallback_price_irr_per_mwh)
        baseline=base*idx
        return np.tile(baseline,(n_iter,months_h)), baseline, pd.DataFrame()

    monthly = aggregate_monthly(daily)
    fit = fit_trend_residuals(monthly)
    resid, months_arr = fit["residuals"], fit["months"]
    last_price, last_month = fit["last_price"], fit["last_month"]
    sigma_hist = fit["sigma_hist"]

    infl_idx = build_inflation_index(months_h, cfg.inflation)
    baseline = last_price * infl_idx

    scale = _calibrate_scale(sigma_hist, cfg.volatility)
    if scale==0.0 or resid.size==0:
        return np.tile(baseline,(n_iter,months_h)), baseline, monthly

    buckets = _residual_buckets(resid, months_arr, cfg.volatility.by_bucket)
    out = np.zeros((n_iter, months_h), dtype=float)

    def future_month_k(k:int)->int:
        return ((last_month - 1 + k) % 12) + 1

    for t in range(months_h):
        if cfg.volatility.by_bucket!="none":
            key = _bucket_key(future_month_k(t+1), cfg.volatility.by_bucket)
        else:
            key = -1
        pool = buckets.get(key, resid)
        eps = _draw_residuals(n_iter, pool, rng, cfg.volatility) * float(scale)
        out[:,t] = baseline[t] * np.exp(eps)

    return out, baseline, monthly

# ========================= __main__ =========================

def _load_yaml_config(path:str)->Dict:
    import yaml
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def _build_price_config(cfg:Dict)->PriceConfig:
    p = cfg.get("price", {})
    daily_path = p.get("iran_daily_path", p.get("iran_daily_csv",""))

    infl_cfg = p.get("inflation", {})
    scenario = infl_cfg.get("scenario","baseline")
    pass_through = float(infl_cfg.get("pass_through",1.0))
    baseline=None; piecewise=None; custom_path=None
    if scenario=="baseline":
        annual=float(infl_cfg.get("baseline",{}).get("annual_rate",0.35))
        baseline=BaselineInflation(annual_rate=annual)
    elif scenario=="piecewise":
        steps=infl_cfg.get("piecewise",[])
        if steps:
            piecewise=[PiecewiseInflationStep(months=int(s.get("months",0)),
                                              annual_rate=float(s.get("annual_rate",0.0))) for s in steps]
    elif scenario=="custom_path":
        mr=infl_cfg.get("custom_path",{}).get("monthly_rates",[])
        if mr: custom_path=CustomPathInflation(monthly_rates=[float(x) for x in mr])
    inflation=InflationConfig(scenario=scenario, pass_through=pass_through,
                              baseline=baseline, piecewise=piecewise, custom_path=custom_path)

    vol_cfg = p.get("volatility", {})
    calib=None
    if "calibrate_sigma" in vol_cfg and vol_cfg["calibrate_sigma"] is not None:
        c=vol_cfg["calibrate_sigma"]
        calib=VolCalib(target_monthly_sigma=(None if c.get("target_monthly_sigma") in [None,"null"] else float(c.get("target_monthly_sigma"))),
                       hard_min=float(c.get("hard_min",0.5)), hard_max=float(c.get("hard_max",2.0)))
    st=None
    if vol_cfg.get("sampler","bootstrap")=="student_t":
        st=VolStudentT(nu=float(vol_cfg.get("student_t",{}).get("nu",5.0)))

    volatility=VolatilityConfig(
        sampler=vol_cfg.get("sampler","bootstrap"),
        by_bucket=vol_cfg.get("by_bucket","month_of_year"),
        scale=float(vol_cfg.get("scale",1.0)),
        calibrate_sigma=calib,
        student_t=st
    )

    return PriceConfig(
        iran_daily_path=daily_path,
        use_vwap=bool(p.get("use_vwap",True)),
        fallback_price_irr_per_mwh=float(p.get("fallback_price_irr_per_mwh",20_000_000)),
        volatility=volatility,
        inflation=inflation
    )

def _jalali_month_range(start_y:int,start_m:int,end_y:int,end_m:int)->List[pd.Timestamp]:
    out=[]
    if not HAS_JDATE:
        months=(end_y-start_y)*12+(end_m-start_m)+1
        start=pd.Timestamp.today().normalize().replace(day=1)
        return list(pd.date_range(start=start, periods=months, freq="ME"))
    y,m=start_y,start_m
    while (y<end_y) or (y==end_y and m<=end_m):
        g=jdatetime.date(y,m,1).togregorian()
        ts=pd.Timestamp(g.year,g.month,1)+pd.offsets.MonthEnd(0)
        out.append(ts)
        m=1 if m==12 else m+1
        y= y+1 if m==1 else y
    return out

def _percentiles_matrix(paths:np.ndarray, qs:List[float])->np.ndarray:
    return np.percentile(paths, qs, axis=0)

if __name__=="__main__":
    CODEBASE=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    CONFIG_PATH=os.path.join(CODEBASE,"config.yaml")

    cfg_yaml=_load_yaml_config(CONFIG_PATH)
    price_cfg=_build_price_config(cfg_yaml)

    # 1404/01 → 1414/12
    start_jy,start_jm=1404,1
    end_jy,end_jm=1414,12
    T=(end_jy-start_jy)*12+(end_jm-start_jm)+1
    n_iter=1000
    seed=int(cfg_yaml.get("monte_carlo",{}).get("random_seed",42))
    rng=np.random.default_rng(seed)

    paths, baseline, monthly_hist = sample_price_path_monthly(
        n_iter=n_iter, months_h=T, rng=rng, cfg=price_cfg
    )
    x_dates=_jalali_month_range(start_jy,start_jm,end_jy,end_jm)

    # Figure 0: Historical monthly series
    if monthly_hist is not None and not monthly_hist.empty:
        plt.figure(figsize=(12,4))
        plt.plot(monthly_hist["date"], monthly_hist["price"], label="Historical monthly (IRR/MWh)")
        plt.title("Historical Monthly Energy Prices (IRR/MWh) — بورس انرژی")
        plt.xlabel("Month (Gregorian)")
        plt.ylabel("Price (IRR/MWh)")
        plt.legend()
        f0=os.path.join(CODEBASE,"price_hist_monthly.png")
        plt.tight_layout(); plt.savefig(f0, dpi=150)

    # Figure 1: Fan chart
    qs=[5,50,95]; qmat=_percentiles_matrix(paths, qs)
    plt.figure(figsize=(12,4))
    plt.plot(x_dates, baseline, label="Baseline (inflation only)")
    plt.plot(x_dates, qmat[1], label="P50")
    plt.fill_between(x_dates, qmat[0], qmat[2], alpha=0.2, label="P5–P95")
    plt.title("Monthly Energy Price Paths (IRR/MWh) — Inflation Baseline + Stochastic Fan")
    plt.xlabel("Month (Gregorian)")
    plt.ylabel("Price (IRR/MWh)")
    plt.legend()
    f1=os.path.join(CODEBASE,"price_sampler_fan_chart.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150)

    # Figure 2: Heatmap
    qs_dense=list(range(5,100,5))
    qmat_dense=_percentiles_matrix(paths, qs_dense)
    plt.figure(figsize=(12,5))
    plt.imshow(qmat_dense, aspect="auto", origin="lower")
    plt.title("Percentile Heatmap of Simulated Monthly Prices (IRR/MWh)")
    plt.xlabel("Time (months 1404→1414)")
    plt.ylabel("Percentile (5→95)")
    xticks=np.linspace(0, T-1, 12, dtype=int)
    plt.xticks(xticks, [x_dates[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
    yticks=np.arange(len(qs_dense))
    plt.yticks(yticks, [str(q) for q in qs_dense])
    f2=os.path.join(CODEBASE,"price_sampler_heatmap.png")
    plt.tight_layout(); plt.savefig(f2, dpi=150)

    print("Saved:", f0 if 'f0' in locals() else "(no historical plot)")
    print("Saved:", f1)
    print("Saved:", f2)
