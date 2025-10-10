
# Weibull (monthly) builder from Open‑Meteo (2004–2024)

This tool fetches **hourly wind_speed_10m** from Open‑Meteo Historical API for 10 Iranian cities (2004‑01‑01..2024‑12‑31), then computes **monthly Weibull parameters (k, c)** by method‑of‑moments and writes `weibull_monthly.csv` compatible with your project.

## How to run
```bash
pip install -r requirements.txt
python build_weibull_from_openmeteo.py --cities_csv cities_iran_10.csv --start_year 2004 --end_year 2024
```
Output: `weibull_monthly.csv` with columns:
```
city, month, count, k, c, mean_ws, std_ws, status, file, location_id
```

## Cities (lat, lon) & sources
- BandarAbbas: 27.183708, 56.277447  (latlong.net)
- Bushehr:     28.983300, 50.816700  (latitude.to / geodatos)
- Kerman:      30.283210, 57.078790  (geodatos.net)
- Zabol:       31.030600, 61.494900  (latitude.to)
- Yazd:        31.897423, 54.356857  (latlong.net)
- Khaf:        34.569897, 60.098819  (database.earth / Mapcarta)
- Tehran:      35.694400, 51.421500  (latitude.to)
- Qazvin:      36.269363, 50.003201  (latlong.net)
- Manjil:      36.744850, 49.400030  (Mapcarta / Iranica)
- Tabriz:      38.066666, 46.299999  (latlong.net)

## Notes
- API: https://open-meteo.com/en/docs/historical-weather-api
- Variables: `hourly=wind_speed_10m`, `wind_speed_unit=ms`, `timezone=UTC`
- Fitting formulae: Method‑of‑moments (Justus et al. 1978); review Appl. Meteor. (1984).
- If a month has very low wind (mean<=0 or std<=0), row flagged `status=LOW_WS` and k,c set NaN.

## License
For **non‑commercial** use in your academic project.
