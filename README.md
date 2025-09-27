# Wind Power Plant Economic & Risk Analysis

📌 **هدف پروژه**  
این ریپازیتوری یک چارچوب محاسباتی برای **امکان‌سنجی و تحلیل ریسک نیروگاه بادی** (مقیاس ~۱۰ MW) فراهم می‌کند.  
به‌جای مقادیر قطعی، خروجی‌ها به‌صورت **توزیعی و احتمالاتی** ارائه می‌شوند (P50/P90، توزیع NPV/IRR/LCOE، سنجه‌های VaR/CVaR، Omega-LCOE).  

---

## ✨ Features

- **مدل‌سازی تولید انرژی بادی** با توزیع‌های ویبول ماهانه (Weibull) و منحنی توان توربین (WTPC).  
- **کالیبراسیون SCADA** برای نزدیک‌کردن مدل به رفتار واقعی سایت.  
- **مدل‌سازی قیمت برق** در سناریوهای داخلی (خرید تضمینی / PPA) و صادراتی.  
- **شبیه‌سازی مونت‌کارلو (MCS)** با ≈۱۰,۰۰۰ تکرار برای انتشار عدم‌قطعیت‌ها.  
- **شاخص‌های اقتصادی:** NPV، IRR، Payback، LCOE.  
- **شاخص‌های ریسک:** VaR، CVaR، Omega-LCOE.  
- **تحلیل حساسیت (Sobol/ANN/SHAP)** برای شناسایی محرک‌های کلیدی.  
- **پشتیبانی از سناریوهای قرارداد و سیاست‌گذاری** (FIT، PPA، صادرات، ریسک FX/تورم).  
- **Pipeline ماژولار و بازتولیدپذیر** با تنظیمات شفاف در فایل `config.yaml`.

---

## 🏗️ Project Architecture

```
data/               # داده‌های خام و پردازش‌شده (Open-Meteo, SCADA, Weibull monthly, ...)
scraping/           # اسکریپت‌های استخراج داده (Open-Meteo API, ...)
modeling/           # ماژول‌های فنی (weibull_fit, turbine_power_curve, ...)
src/
  ├─ wind_resource.py   # مدل باد و تولید انرژی
  ├─ price_model.py     # مدل‌سازی قیمت برق
  ├─ cashflow.py        # محاسبه جریان نقدی و شاخص‌های اقتصادی
  ├─ monte_carlo.py     # شبیه‌سازی مونت‌کارلو و ادغام عدم‌قطعیت‌ها
  ├─ report.py          # تولید گزارش، نمودارها و خروجی HTML/Excel
config.yaml          # فایل پیکربندی مرکزی (ورودی‌ها، پارامترها، سناریوها)
requirements.txt     # وابستگی‌های پایتون
reports/             # گزارش‌های خروجی (Markdown, HTML, PDF, PNG, CSV/Excel)
notebooks/           # نوت‌بوک‌های توسعه و اعتبارسنجی
```

---

## 📊 Data Sources

- **Open-Meteo ERA5 API** → داده‌های ساعتی باد/دما/فشار در ارتفاع ۱۰۰ متر (برای ۱۰ شهر منتخب).  
- **Weibull Monthly Dataset** → پارامترهای (k,c) ماهانه برای هر شهر.  
- **SCADA Dataset** → برای کالیبراسیون منحنی توان.  
- **Electricity Prices (U.S./Regional)** → برای ساخت سناریوهای قیمتی.  

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>

# Create environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # (Linux/macOS)
.venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

1. **تنظیم ورودی‌ها در `config.yaml`**  
   - مکان (شهر، مختصات)  
   - ظرفیت نیروگاه (MW)  
   - پارامترهای باد (k,c ماهانه)  
   - منحنی توان توربین  
   - CAPEX، OPEX، نرخ تنزیل  
   - سناریوی قیمت (FIT / صادرات)  

2. **اجرای تحلیل:**

```bash
python src/monte_carlo.py
```

3. **خروجی‌ها:**
   - `reports/summary.md` → خلاصه مدیریتی  
   - `reports/plots/` → نمودارها (PNG)  
   - `reports/cashflow.xlsx` → جریان نقدی و KPIها  
   - `reports/output.html` → گزارش تعاملی  

---

## 📈 Outputs

- Annual production, revenue, OPEX, cashflow charts  
- Histogram & CDF of NPV / IRR  
- Tornado sensitivity diagram  
- Scenario heatmap (Price × Production)  
- Executive summary (Markdown/HTML)  

---

## 🧩 Methodology (High-level)

1. **Wind → Power → Energy:**  
   - نمونه‌گیری سرعت باد ماهانه از Weibull(k,c).  
   - نگاشت به توان با WTPC.  
   - اعمال تلفات و دسترس‌پذیری.  

2. **Price Modeling:**  
   - مسیرهای تصادفی قیمت (ARIMA/Regime Switching).  
   - سناریوهای قراردادی (FIT، صادرات).  

3. **Cashflow & Economics:**  
   - درآمد = انرژی × قیمت.  
   - کسر CAPEX + OPEX.  
   - محاسبه NPV، IRR، Payback، LCOE.  

4. **Risk Simulation (Monte Carlo):**  
   - ۱۰٬۰۰۰ مسیر شبیه‌سازی.  
   - گزارش توزیع P50/P90، VaR/CVaR.  

---

## 📌 Notes

- تمام واحدها در محور/ستون‌ها شفاف درج می‌شوند (MW, MWh, IRR%, USD/IRR).  
- همه‌ی ورودی‌ها و پارامترها در `config.yaml` نگهداری می‌شوند → بازتولید و سناریوسازی آسان.  
- ماژول‌ها کاملاً مستقل هستند و امکان توسعه آینده (Curtailment، باتری، قراردادهای پیچیده‌تر) وجود دارد.  

---

## 👥 Authors

- **Research & Code:** [Your Name / Team]  
- **Supervisor / Advisor:** [Name if academic]  

---

## 📄 License

MIT License – feel free to use, modify, and distribute with attribution.
