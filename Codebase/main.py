#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Deterministic cost estimator (MVP) for an onshore wind farm.

What it does (now):
- Prompts you for the minimum inputs.
- Computes:
    * CAPEX_total_usd  = capacity_mw * 1000 * capex_usd_per_kw
    * OPEX_year1_usd   = capacity_mw * 1000 * opex_usd_per_kw_yr
    * PV_OPEX_usd      = present value of the inflated OPEX stream over project life
    * NPV_costs_usd    = CAPEX_total_usd + PV_OPEX_usd
- Builds a loan amortization schedule for information (debt service),
  but DOES NOT add debt service to NPV_costs to avoid double counting.

What it does NOT do (yet):
- No revenues, no taxes, no depreciation, no uncertainty.
  These will be added after we lock the MVP costs.
"""

from typing import List, Dict


def prompt_float(prompt_txt: str, default: float = None) -> float:
    """Prompt user for a float with an optional default; re-prompt on invalid input."""
    while True:
        tip = f" [default: {default}]" if default is not None else ""
        raw = input(f"{prompt_txt}{tip}: ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  → Invalid number, please try again.")


def annuity_payment(principal: float, rate: float, n_years: int) -> float:
    """Fixed annual payment for a standard amortizing loan."""
    if n_years <= 0:
        return 0.0
    if rate == 0.0:
        return principal / n_years
    return principal * rate / (1.0 - (1.0 + rate) ** (-n_years))


def build_debt_schedule(principal: float, loan_rate: float, n_years: int) -> List[Dict[str, float]]:
    """Yearly amortization schedule: payment, interest, principal, remaining."""
    schedule = []
    payment = annuity_payment(principal, loan_rate, n_years)
    remaining = principal
    for year in range(1, n_years + 1):
        interest = remaining * loan_rate
        principal_paid = payment - interest
        remaining = max(0.0, remaining - principal_paid)
        schedule.append({
            "year": year,
            "payment": payment,
            "interest": interest,
            "principal": principal_paid,
            "remaining": remaining,
        })
    return schedule


def present_value(cashflows: List[float], discount_rate: float) -> float:
    """Present value (end-of-year convention) of CF[1..N] discounted at 'discount_rate'."""
    pv = 0.0
    for t, cf in enumerate(cashflows, start=1):
        pv += cf / ((1.0 + discount_rate) ** t)
    return pv


def main() -> None:
    print("=== Wind Farm Cost MVP (Deterministic) ===\n")
    print("Please enter rates as decimals. For example, 0.12 means 12%.")

    # --- Core inputs ---
    capacity_mw = prompt_float("Plant capacity (MW)")
    years = int(prompt_float("Project lifetime (years)", 20))

    # Discount rate: the rate to discount future cash flows to present value.
    discount_rate = prompt_float(
        "Discount rate (decimal, e.g., 0.12 = 12%) — reflects required return / risk",
        0.12
    )

    # Inflation used to index O&M over time (not CAPEX).
    inflation = prompt_float(
        "Annual O&M inflation (decimal, e.g., 0.20 = 20%)",
        0.20
    )

    # Financing (for information): how much of CAPEX is financed with debt, and the loan rate.
    debt_ratio = prompt_float(
        "Debt ratio (0..1). Use 1.0 if the entire CAPEX is debt-financed",
        1.0
    )
    loan_rate = prompt_float(
        "Loan interest rate (decimal, e.g., 0.18 = 18%)",
        0.18
    )

    # Benchmarks (override if you have better local values)
    capex_usd_per_kw = prompt_float(
        "CAPEX (USD per kW) — capital cost per installed kW",
        1154.0
    )
    opex_usd_per_kw_yr = prompt_float(
        "OPEX (USD per kW per year) — fixed annual O&M per kW",
        43.0
    )

    # --- Compute deterministic costs ---
    capex_total_usd = capacity_mw * 1000.0 * capex_usd_per_kw
    opex_year1_usd = capacity_mw * 1000.0 * opex_usd_per_kw_yr

    # Build an indexed OPEX stream (end-of-year convention)
    opex_stream = [opex_year1_usd * ((1.0 + inflation) ** (t - 1)) for t in range(1, years + 1)]
    pv_opex = present_value(opex_stream, discount_rate)

    # Debt service (informational only; not added into NPV_costs)
    debt_principal = capex_total_usd * debt_ratio
    debt_sched = build_debt_schedule(debt_principal, loan_rate, years)
    pv_debt_service = present_value([row["payment"] for row in debt_sched], discount_rate)

    # NPV of costs = CAPEX (time 0) + PV(OPEX over lifetime)
    npv_costs = capex_total_usd + pv_opex

    # --- Print results ---
    print("\n--- Results (USD) ---")
    print(f"CAPEX_total_usd:                {capex_total_usd:,.2f}")
    print(f"OPEX_year1_usd:                 {opex_year1_usd:,.2f}")
    print(f"PV_OPEX_usd (discounted):       {pv_opex:,.2f}")
    print(f"NPV_costs_usd (CAPEX + PV OPEX): {npv_costs:,.2f}")

    print("\n--- Debt Service (informational; not included in NPV_costs) ---")
    print(f"Debt_principal_usd:             {debt_principal:,.2f}")
    print(f"Loan_rate:                      {loan_rate:.4f}   |  Years: {years}")
    print(f"PV_Debt_Service_usd @discount:  {pv_debt_service:,.2f}")

    if years <= 6:
        for row in debt_sched:
            print(f"Year {row['year']:2d} | Payment: {row['payment']:>12,.2f} | "
                  f"Interest: {row['interest']:>10,.2f} | Principal: {row['principal']:>10,.2f} | "
                  f"Remaining: {row['remaining']:>12,.2f}")
    else:
        first = debt_sched[0]
        mid = debt_sched[len(debt_sched)//2]
        last = debt_sched[-1]
        print(f"Y{first['year']:2d} Payment: {first['payment']:>12,.2f} | "
              f"Interest: {first['interest']:>10,.2f} | Principal: {first['principal']:>10,.2f} | "
              f"Remaining: {first['remaining']:>12,.2f}")
        print(f"Y{mid['year']:2d} Payment: {mid['payment']:>12,.2f} | "
              f"Interest: {mid['interest']:>10,.2f} | Principal: {mid['principal']:>10,.2f} | "
              f"Remaining: {mid['remaining']:>12,.2f}")
        print(f"Y{last['year']:2d} Payment: {last['payment']:>12,.2f} | "
              f"Interest: {last['interest']:>10,.2f} | Principal: {last['principal']:>10,.2f} | "
              f"Remaining: {last['remaining']:>12,.2f}")

    print("\nNotes:")
    print("- NPV of costs is defined here as CAPEX (time 0) + the present value of indexed OPEX.")
    print("- Debt service is shown for transparency but excluded from NPV of costs to avoid double counting.")
    print("- We will add revenues, taxes, depreciation, and uncertainty in the next steps.")
    print("- Units: USD; if you prefer IRR or mixed currency, we can add an FX layer later.")


if __name__ == "__main__":
    main()
