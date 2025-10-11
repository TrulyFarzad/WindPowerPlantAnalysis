# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from backend_adapter import (
    load_default_config, make_cfg_for_run, run_project_with_config
)
import traceback
import json

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="Wind Plant — Economic MVP WebUI", layout="wide")
st.title("🎛️ Wind Plant Project — Web UI")
st.caption("ورودی‌ها را تنظیم کنید، سپس «محاسبه» را بزنید؛ گزارش HTML زیر همین صفحه ظاهر می‌شود.")

# --------------------------- Helpers ---------------------------
def normalize_root(root_str: str) -> Path:
    p = Path(root_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"مسیر یافت نشد: {p}")
    return p

def load_cities_from_weibull(project_root: Path):
    candidates = [
        project_root / "weibull_monthly.csv",
        project_root / "data" / "wind" / "weibull_monthly.csv",
        Path("weibull_monthly.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "city" in df.columns:
                return sorted(df["city"].dropna().unique().tolist())
    return ["Manjil","Khaf","Zabol","BandarAbbas","Bushehr","Kerman","Yazd","Tehran","Qazvin","Tabriz"]

def load_turbines_from_csv(project_root: Path):
    candidates = [
        project_root / "windpowerlib_turbine_library.csv",
        project_root / "data" / "turbines" / "windpowerlib_turbine_library.csv",
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

# --------------------------- Sidebar: Project root ---------------------------
with st.sidebar:
    st.header("📁 مسیر پروژه")
    project_root_input = st.text_input(
        "ریشهٔ پروژه (پوشهٔ WindPowerPlantAnalysis)",
        value=str(Path.cwd())
    )
    st.info("اگر این وب‌اپ را داخل همان پوشهٔ پروژه اجرا کرده‌اید، همین کافی است.")

# --------------------------- Bootstrap config ---------------------------
try:
    root = normalize_root(project_root_input)
    default_cfg = load_default_config(root)
    cfg_ok = True
except Exception as e:
    cfg_ok = False
    st.error(f"عدم موفقیت در آماده‌سازی پروژه: {e}")

if not cfg_ok:
    st.stop()

cities   = load_cities_from_weibull(root)
turbines = load_turbines_from_csv(root)

# --------------------------- Inputs ---------------------------
st.subheader("⚙️ تنظیمات ورودی")
colA, colB = st.columns([1,1])

with colA:
    all_cities = st.checkbox("برای همهٔ شهرها اجرا شود", value=False)
    city_default = default_cfg.get("site",{}).get("city", default_cfg.get("wind",{}).get("city","Khaf"))
    city_index = cities.index(city_default) if city_default in cities else 0
    city = st.selectbox("شهر", options=cities, index=city_index, disabled=all_cities)
    st.caption("اگر «همهٔ شهرها» فعال باشد، انتخاب شهر غیرفعال می‌شود.")

    turbine_source = st.selectbox("منبع منحنی توان", options=["windpowerlib","iec_like"], index=0)
    turb_default   = default_cfg.get("turbine",{}).get("name", default_cfg.get("wind",{}).get("turbine_name","V112/3000"))
    turb_index     = turbines.index(turb_default) if turb_default in turbines else 0
    turbine_name   = st.selectbox("نام توربین (windpowerlib)", options=turbines, index=turb_index, disabled=(turbine_source!="windpowerlib"))
    iec_class      = st.selectbox("مدل IEC (کلاس باد/آشفتگی)", options=["IA","IB","IIA","IIB","IIIA","IIIB"], index=2, disabled=(turbine_source!="iec_like"))

    years       = st.number_input("افق شبیه‌سازی (سال)", min_value=1, max_value=50, value=int(default_cfg.get("project",{}).get("years",10)), step=1)
    capacity_mw = st.number_input("ظرفیت مزرعه (MW)", min_value=1.0, max_value=1000.0, value=float(default_cfg.get("plant",{}).get("capacity_mw",20.0)), step=1.0)

with colB:
    n_scen   = st.number_input("تعداد سناریوهای مونت‌کارلو", min_value=100, max_value=100000, value=int(default_cfg.get("project",{}).get("n_scenarios", default_cfg.get("monte_carlo",{}).get("iterations",2000))), step=100)
    discount = st.number_input("نرخ تنزیل سالانه", min_value=0.0, max_value=1.0, value=float(default_cfg.get("project",{}).get("discount_rate",0.12)), format="%.3f")

    # برای سازگاری، اول economics، در غیر این صورت costs
    capex_default = default_cfg.get("economics",{}).get("capex_usd_per_kw",
                     default_cfg.get("costs",{}).get("capex_usd_per_kw", 1000.0))
    opex_default  = default_cfg.get("economics",{}).get("opex_usd_per_kw_yr",
                     default_cfg.get("costs",{}).get("opex_usd_per_kw_yr", 40.0))
    infl_default  = default_cfg.get("economics",{}).get("inflation_opex",
                     default_cfg.get("costs",{}).get("inflation_opex", 0.20))

    capex    = st.number_input("CAPEX (USD/kW)", min_value=0.0, max_value=5000.0, value=float(capex_default), step=10.0)
    opex     = st.number_input("OPEX سالانه (USD/kW-yr)", min_value=0.0, max_value=500.0, value=float(opex_default), step=1.0)
    infl     = st.number_input("تورم OPEX سالانه", min_value=0.0, max_value=1.0, value=float(infl_default), format="%.3f")

st.divider()
st.subheader("🛠️ گزینه‌های پیشرفته")
optimize_capacity = st.checkbox("بهینه‌سازی ظرفیت (grid-search ساده روی ظرفیت)")
capacity_grid     = st.text_input("شبکهٔ ظرفیت‌ها برای بهینه‌سازی (MW)", value="10,15,20,25,30")
st.caption("اگر فعال باشد، برای هر شهر ظرفیت‌های لیست‌شده تست می‌شود و بهترین (NPV P50) انتخاب می‌شود؛ اگر KPI JSON موجود نباشد، اولین اجرای موفق انتخاب می‌شود.")

st.divider()

run = st.button("🚀 محاسبه")
if not run:
    st.stop()

# --------------------------- Run ---------------------------
with st.spinner("⏳ در حال محاسبه…"):
    try:
        results = []
        selected_cities = cities if all_cities else [city]

        # نوار پیشرفت برای شهرها
        city_progress = st.progress(0, text="پیشرفت شهرها")
        total_cities = len(selected_cities)

        for ci, c in enumerate(selected_cities, start=1):
            st.markdown(f"#### 🔄 اجرای شهر: **{c}**")
            city_area = st.container()
            with city_area:
                if optimize_capacity:
                    st.write("بهینه‌سازی ظرفیت در حال اجرا…")
                else:
                    st.write(f"ظرفیت انتخابی: {capacity_mw} MW")

                caps = [t.strip() for t in capacity_grid.split(",") if t.strip()] if optimize_capacity else []
                cap_progress = None
                if optimize_capacity and len(caps) > 0:
                    cap_progress = st.progress(0, text="پیشرفت ظرفیت‌ها")

                chosen_capacity = capacity_mw
                final_html = None

                if optimize_capacity:
                    best_cap = None
                    best_html = None
                    best_score = None
                    total_caps = len(caps)
                    for idx, token in enumerate(caps, start=1):
                        try:
                            cap_val = float(token)
                        except:
                            continue

                        cfg = make_cfg_for_run(default_cfg, c, years, cap_val, discount, capex, opex, infl,
                                               turbine_source, turbine_name, iec_class, n_scen)

                        out_html = run_project_with_config(root, cfg)

                        # KPI کنار HTML اگر باشد، بخوان
                        score = None
                        kpi_json = Path(out_html).with_suffix(".json")
                        if kpi_json.exists():
                            try:
                                data = json.loads(kpi_json.read_text(encoding="utf-8"))
                                score = data.get("kpis",{}).get("npv_p50", None)
                            except Exception:
                                score = None

                        if best_score is None and score is None:
                            best_cap, best_html = cap_val, out_html
                        elif (score is not None) and (best_score is None or score > best_score):
                            best_cap, best_html, best_score = cap_val, out_html, score

                        if cap_progress:
                            cap_progress.progress(min(idx/total_caps,1.0), text=f"پیشرفت ظرفیت‌ها ({idx}/{total_caps})")

                    chosen_capacity = best_cap if best_cap is not None else capacity_mw
                    final_html = best_html
                else:
                    cfg = make_cfg_for_run(default_cfg, c, years, capacity_mw, discount, capex, opex, infl,
                                           turbine_source, turbine_name, iec_class, n_scen)
                    final_html = run_project_with_config(root, cfg)

                results.append({"city": c, "capacity": chosen_capacity, "html": final_html})

            city_progress.progress(min(ci/total_cities,1.0), text=f"پیشرفت شهرها ({ci}/{total_cities})")

        st.success("محاسبه کامل شد ✅")

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
