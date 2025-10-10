
import streamlit as st
import pandas as pd
from pathlib import Path
from backend_adapter import (
    find_project_root, load_default_config, make_cfg_for_run, run_project_with_config
)
import traceback


st.set_page_config(page_title="Wind Plant — Economic MVP WebUI", layout="wide")
st.title("🎛️ Wind Plant Project — Web UI")
st.caption("ورودی‌ها را تنظیم کنید، سپس «محاسبه» را بزنید؛ گزارش HTML زیر همین صفحه ظاهر می‌شود.")

# ---- Sidebar: project root ----
with st.sidebar:
    st.header("📁 مسیر پروژه")
    project_root = st.text_input(
        "Project root (پوشه‌ای که main.py داخلش است)",
        value=str(Path.cwd())
    )
    st.info("اگر این وب‌اپ را داخل همان پوشهٔ پروژه اجرا کرده‌اید، همین کافی است.")

# ---- bootstrap config ----
try:
    root = find_project_root(project_root)
    default_cfg = load_default_config(root)
    cfg_ok = True
except Exception as e:
    cfg_ok = False
    st.error(f"Couldn't locate project: {e}")

def load_cities_from_weibull(project_root: str):
    candidates = [
        Path(project_root) / "weibull_monthly.csv",
        Path(project_root) / "data" / "wind" / "weibull_monthly.csv",
        Path("weibull_monthly.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "city" in df.columns:
                return sorted(df["city"].dropna().unique().tolist())
    return ["Manjil","Khaf","Zabol","BandarAbbas","Bushehr","Kerman","Yazd","Tehran","Qazvin","Tabriz"]

def load_turbines_from_csv(project_root: str):
    candidates = [
        Path(project_root) / "windpowerlib_turbine_library.csv",
        Path(project_root) / "data" / "turbines" / "windpowerlib_turbine_library.csv",
        Path("windpowerlib_turbine_library.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            cols = df.columns.str.lower()
            name_col = None
            for c in ["turbine_type","name","type","model"]:
                if c in cols:
                    name_col = df.columns[list(cols).index(c)]
                    break
            if name_col is None:
                name_col = df.columns[0]
            return sorted(df[name_col].dropna().astype(str).unique().tolist())
    return ["V112/3000","V90/3000","N117/3600"]

if not cfg_ok:
    st.stop()

cities   = load_cities_from_weibull(root)
turbines = load_turbines_from_csv(root)

# ---- Inputs ----
st.subheader("⚙️ تنظیمات ورودی")
colA, colB = st.columns([1,1])

with colA:
    all_cities = st.checkbox("برای همهٔ شهرها اجرا شود", value=False)
    city_default = default_cfg.get("site",{}).get("city","Khaf")
    city_index = cities.index(city_default) if city_default in cities else 0
    city = st.selectbox("شهر", options=cities, index=city_index, disabled=all_cities)
    st.caption("اگر «همهٔ شهرها» فعال باشد، انتخاب شهر غیرفعال می‌شود.")

    turbine_source = st.selectbox("منبع منحنی توان", options=["windpowerlib","iec_like"], index=0)
    turb_default   = default_cfg.get("turbine",{}).get("name","V112/3000")
    turb_index     = turbines.index(turb_default) if turb_default in turbines else 0
    turbine_name   = st.selectbox("نام توربین (windpowerlib)", options=turbines, index=turb_index, disabled=(turbine_source!="windpowerlib"))
    iec_class      = st.selectbox("مدل IEC (کلاس باد/آشفتگی)", options=["IA","IB","IIA","IIB","IIIA","IIIB"], index=2, disabled=(turbine_source!="iec_like"))

    years       = st.number_input("افق شبیه‌سازی (سال)", min_value=1, max_value=50, value=int(default_cfg.get("project",{}).get("years",10)), step=1)
    capacity_mw = st.number_input("ظرفیت مزرعه (MW)", min_value=1.0, max_value=1000.0, value=float(default_cfg.get("plant",{}).get("capacity_mw",20.0)), step=1.0)

with colB:
    n_scen   = st.number_input("تعداد سناریوهای مونت‌کارلو", min_value=100, max_value=100000, value=int(default_cfg.get("project",{}).get("n_scenarios",2000)), step=100)
    discount = st.number_input("نرخ تنزیل سالانه", min_value=0.0, max_value=1.0, value=float(default_cfg.get("project",{}).get("discount_rate",0.12)), format="%.3f")
    capex    = st.number_input("CAPEX (USD/kW)", min_value=0.0, max_value=5000.0, value=float(default_cfg.get("economics",{}).get("capex_usd_per_kw",1000.0)), step=10.0)
    opex     = st.number_input("OPEX سالانه (USD/kW-yr)", min_value=0.0, max_value=500.0, value=float(default_cfg.get("economics",{}).get("opex_usd_per_kw_yr",40.0)), step=1.0)
    infl     = st.number_input("تورم OPEX سالانه", min_value=0.0, max_value=1.0, value=float(default_cfg.get("economics",{}).get("inflation_opex",0.20)), format="%.3f")

st.divider()
st.subheader("🛠️ گزینه‌های پیشرفته")
optimize_capacity = st.checkbox("بهینه‌سازی ظرفیت (grid-search ساده روی ظرفیت)")
capacity_grid     = st.text_input("شبکهٔ ظرفیت‌ها برای بهینه‌سازی (MW)", value="10,15,20,25,30")
st.caption("اگر فعال باشد، برای هر شهر ظرفیت‌های لیست‌شده تست می‌شود و بهترین (NPV P50) انتخاب می‌شود؛ اگر KPI JSON موجود نباشد، اولین اجرای موفق انتخاب می‌شود.")

st.divider()
run = st.button("🚀 محاسبه")

if not run:
    st.stop()

with st.spinner("در حال اجرا…"):
    try:
        results = []
        selected_cities = cities if all_cities else [city]

        for c in selected_cities:
            chosen_capacity = capacity_mw
            final_html = None

            if optimize_capacity:
                best_cap = None
                best_html = None
                best_score = None
                for token in [t.strip() for t in capacity_grid.split(",") if t.strip()]:
                    try:
                        cap_val = float(token)
                    except:
                        continue
                    cfg = make_cfg_for_run(default_cfg, c, years, cap_val, discount, capex, opex, infl,
                                           turbine_source, turbine_name, iec_class, n_scen)
                    out_html = run_project_with_config(root, cfg)

                    # اختیاری: KPI کنار HTML اگر باشد، بخوان
                    score = None
                    kpi_json = Path(out_html).with_suffix(".json")
                    if kpi_json.exists():
                        try:
                            import json
                            data = json.loads(kpi_json.read_text(encoding="utf-8"))
                            score = data.get("kpis",{}).get("npv_p50", None)
                        except Exception:
                            score = None

                    if best_score is None and score is None:
                        best_cap, best_html = cap_val, out_html
                    elif (score is not None) and (best_score is None or score > best_score):
                        best_cap, best_html, best_score = cap_val, out_html, score

                chosen_capacity = best_cap if best_cap is not None else capacity_mw
                final_html = best_html
            else:
                cfg = make_cfg_for_run(default_cfg, c, years, capacity_mw, discount, capex, opex, infl,
                                       turbine_source, turbine_name, iec_class, n_scen)
                final_html = run_project_with_config(root, cfg)

            results.append({"city": c, "capacity": chosen_capacity, "html": final_html})

        st.success("اجرا کامل شد. گزارش‌ها در زیر نمایش داده می‌شوند.")
        for r in results:
            st.markdown(f"### 📄 گزارش شهر: {r['city']} — ظرفیت: {r['capacity']} MW")
            try:
                html = Path(r["html"]).read_text(encoding="utf-8")
                st.components.v1.html(html, height=900, scrolling=True)
            except Exception as e:
                st.warning(f"نتوانستم HTML را نمایش دهم ({e}). مسیر فایل: {r['html']}")
    except Exception as e:
        st.error(f"💥 خطا در اجرا: {e}")
        st.code(traceback.format_exc())
