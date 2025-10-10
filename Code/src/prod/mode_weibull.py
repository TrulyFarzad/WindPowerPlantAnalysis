# prod/mode_weibull.py  (گزیده‌ی تغییرها – نقاط الحاق)
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any

# ... بقیه ایمپورت‌ها ...
# فرض: از قبل generate_hourly_from_weibull(...) داشتیم که v_hourly (len=~720) می‌دهد

def apply_diurnal_factor(
    v_hourly: np.ndarray,
    month_idx0: int,           # 0..11
    hours_idx: np.ndarray,     # آرایه ساعت‌های 0..23 با طول len(v_hourly)
    diurnal: Optional[np.ndarray]  # (12,24) یا (24,)
) -> np.ndarray:
    if diurnal is None:
        return v_hourly
    if diurnal.ndim == 2:   # monthly × hour
        f = diurnal[month_idx0, hours_idx]
    else:
        f = diurnal[hours_idx]
    v_mod = v_hourly * f
    # حفظ میانگین ماهانه (no-bias): باز-نرمال‌سازی به میانگین قبلی
    mu0 = v_hourly.mean()
    mu1 = v_mod.mean()
    if mu1 > 0:
        v_mod = v_mod * (mu0 / mu1)
    return v_mod

def simulate_month(
    rng, k: float, c_at_hub: float, samples_per_month: int, 
    month_idx0: int, ar1_phi: float, 
    diurnal: Optional[np.ndarray],  # اضافه شد
    **kwargs
) -> np.ndarray:
    """
    خروجی: v_hourly با طول samples_per_month
    """
    # 1) نمونه‌گیری ویبول + OU/AR(1) روی log-speed (کد موجود شما)
    v = _hourly_weibull_with_ar1(rng, k, c_at_hub, samples_per_month, ar1_phi)
    # 2) index ساعت‌ها برای همان ماه
    hours = np.arange(samples_per_month) % 24
    # 3) تزریق دیورنال + re-normalize ماهانه
    v = apply_diurnal_factor(v, month_idx0, hours, diurnal)
    return v

# در جایی که حلقه‌ی ماه‌ها اجرا می‌شود:
# diurnal_profile می‌تواند None یا (12,24) یا (24,) باشد
def run_mode_weibull_with_options(*args, diurnal_profile: Optional[np.ndarray] = None, **kwargs):
    """Placeholder wrapper; to be implemented or wired up to existing simulator."""
    pass
    # ...
    for t in range(T):  # هر ماه
        mi = t % 12
        v_hour = simulate_month(
            rng, k_month[mi], c_month_hub[mi], samples_per_month,
            month_idx0=mi, ar1_phi=ar1_phi, diurnal=diurnal_profile
        )
        # سپس v→P با منحنی توان + TI smoothing + hysteresis (کد موجود شما)
        # بعد جمع ساعتی → MWh ماهانه
    # ...
