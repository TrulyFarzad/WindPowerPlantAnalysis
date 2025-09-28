# Wind Plant Economic Analysis under Uncertainty

![Wind Energy](flowchart.jpg)

## 📌 Overview
This project provides a **comprehensive probabilistic framework** for the **economic analysis of wind power plants**.  
Unlike traditional deterministic models, we use **Monte Carlo Simulation (MCS)** with correlated uncertainties to evaluate **production, revenues, and financial indicators (NPV, IRR, LCOE, VaR/CVaR, Omega-LCOE)**.

The architecture integrates:
- **Wind Resource Modeling**: Monthly Weibull parameters, vertical extrapolation, air density adjustment.
- **Power Curve Modeling**: Manufacturer curves + **SCADA calibration** to align with real-world turbine behavior.
- **Electricity Price Modeling**: Historical datasets + block-bootstrap resampling to capture seasonal/market risks.
- **Economic Model**: CAPEX, OPEX, contracts (PPA/market/export), financing, taxes, and inflation scenarios.
- **Risk Metrics**: Probabilistic KPIs with emphasis on **risk-adjusted returns**.

This framework enables **robust investment decisions** under uncertainty in wind energy projects.

---

## 🏗️ Project Structure
```
Code.zip/                       # Source code (Python)
│── main.py                     # Main entry point (runs full pipeline)
│── production_model.py         # Wind production model (Modes A/B/C)
│── price_model.py              # Electricity price simulation
│── economics.py                # Cashflow + financial KPIs
│── reporting.py                # Report generation (HTML/CSV/plots)
│── config.yaml                 # Central configuration file
│
├── data/                       # Datasets
│   ├── weibull_monthly.csv     # Monthly Weibull parameters (per city)
│   ├── wind_turbine_scada/     # SCADA dataset for calibration
│   ├── wind_power_forecasting/ # Time-series forecasting dataset
│   ├── us_electricity_prices/  # U.S. electricity prices dataset
│
├── outputs/                    # Simulation results
│   ├── mvp_report.html         # Main Monte Carlo simulation report
│   ├── production_paths.csv    # Energy production samples
│   ├── price_paths.csv         # Electricity price samples
│   ├── npv_distribution.png    # Example output chart
│
└── README.md                   # (this file)
```

---

## 🔑 Key Features
- **Three Production Modes (A/B/C):**
  - **Mode A**: Weibull-only (statistical baseline)
  - **Mode B**: SCADA-calibrated (realistic turbine behavior)
  - **Mode C**: Hybrid (adds diurnal/short-term profiles)
- **Monte Carlo Simulation** with thousands of scenarios (default N=2000–10000).
- **Config-driven architecture** (`config.yaml`) ensures reproducibility and transparency.
- **Modular design**: Replace/improve any module (production, price, economics) independently.
- **Outputs for investors**: P50/P90 production, NPV distribution, IRR distribution, VaR/CVaR risk metrics.

---

## 📊 Methodology
1. **Wind Modeling**  
   - Weibull parameters (k, c) estimated per month per city.  
   - Adjusted for hub height and air density.  
   - Integrated with turbine power curves.

2. **Production Modeling**  
   - Converts wind distributions to energy using turbine curves.  
   - Optional SCADA calibration (power scale, v-shift, TI, availability).  
   - Supports diurnal/seasonal profiles.

3. **Price Modeling**  
   - U.S. electricity price dataset (monthly).  
   - Converted to USD/MWh.  
   - Monte Carlo via **block-bootstrap (12-month blocks)** to preserve seasonality.  

4. **Economic Modeling**  
   - Cashflows from CAPEX, OPEX, revenues.  
   - Discounted at configurable project WACC.  
   - KPIs: NPV, IRR, LCOE, Payback, VaR/CVaR, Omega-LCOE.

5. **Reporting**  
   - Outputs consolidated in **mvp_report.html** (tables + charts).  
   - Fan charts for revenue/cashflows.  
   - Histograms for NPV and IRR.  

---

## ⚙️ Installation & Usage
### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/wind-plant-economic-analysis.git
cd wind-plant-economic-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Base Simulation
```bash
python main.py --config config.yaml
```

---

## 📂 Datasets
- **`weibull_monthly.csv`** → Monthly wind speed Weibull parameters (10 cities in Iran).  
- **SCADA Dataset (Kaggle)** → 2018 10-min SCADA data for calibration.  
- **Wind Power Forecasting Dataset (Kaggle)** → 2.5 years, 10-min time series for turbine behavior. 
- **U.S. Electricity Prices (Kaggle)** → Monthly prices from 2001–2024.  

All datasets are **publicly available** and referenced in the project documentation.

---

## 📑 References
- [Windpowerlib Documentation](https://windpowerlib.readthedocs.io/)  
- [Kaggle – Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)  
- [Kaggle – Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting)  
- [Kaggle – U.S. Electricity Prices](https://www.kaggle.com/datasets/nicholasjhana/US-electricity-prices)  
- [IRENA Reports](https://www.irena.org/)  

---

## 👥 Authors
- **فرزاد نورسته**  
- **علی خسروی**  

Advisor: **دکتر حبیب رجبی مشهدی**  
Department of Electrical Engineering (Power Systems), **Ferdowsi University of Mashhad (FUM)**

---

## 📜 License
This project is released under the **CC BY-NC 4.0 License** (Non-Commercial).

---

## ⭐ Acknowledgements
This repository is part of our **undergraduate thesis project**:  
*"تحلیل ریسک و بررسی امکان‌سنجی پروژه‌های صنعتی با تأکید بر پروژه‌های تولید انرژی تجدیدپذیر"*.
