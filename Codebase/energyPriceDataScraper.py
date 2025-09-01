#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IREMA Daily Energy Price Scraper (Playwright, Python)

- URL: https://www.irema.ir/fa/market-data/daily-data/price/average
- Inputs (Jalali): from_date="1392/01/01", to_date="1404/06/09"
- Output: Excel file merged across all pages → Desktop

Requirements (install once):
    pip install playwright pandas openpyxl persiantools
    playwright install chromium

Run examples:
    python irema_playwright_scraper.py
    # or custom
    python irema_playwright_scraper.py --from 1392/01/01 --to 1404/06/09 --out "C:/Users/%USERNAME%/Desktop/irema_prices.xlsx" --max-days-per-batch 1000 --headed

Notes:
- این اسکریپر تاریخ جلالی را همان‌طور که در سایت است ذخیره می‌کند (ستون date_jalali).
- اگر سایت اعداد فارسی نمایش دهد یا جداکنندهٔ «,» داشته باشد، به عدد صحیح (ریال/مگاوات‌ساعت) تبدیل می‌شود.
- اگر سایت روی هر جستجو فقط ~۳۴ صفحه (≈۱۰۳۰ ردیف) را نمایش دهد، اسکریپت به‌صورت خودکار بازهٔ تاریخ را به پارت‌های کوچک‌تر (مثلاً ۱۰۰۰ روزه) می‌شکند و یکی‌یکی جمع می‌کند.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
from datetime import timedelta

import pandas as pd
from persiantools.jdatetime import JalaliDate
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = "https://www.irema.ir/fa/market-data/daily-data/price/average"

# ---- Helpers ---------------------------------------------------------------
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٬", "0123456789,")

def normalize_number(text: str):
    """Convert displayed price like "243,859" or "۲۴۳٬۸۵۹" to int. Empty→None."""
    if text is None:
        return None
    t = text.strip().translate(PERSIAN_DIGITS).replace(",", "")
    return int(t) if t.isdigit() else None


def get_desktop_default(filename: str = "irema_average_prices.xlsx") -> Path:
    # Try $USER/$USERNAME, but fall back to home/Desktop
    user = os.environ.get("USER") or os.environ.get("USERNAME")
    if sys.platform.startswith("win") and user:
        return Path(f"C:/Users/{user}/Desktop/{filename}")
    return Path.home() / "Desktop" / filename


def parse_jalali(s: str) -> JalaliDate:
    return JalaliDate.strptime(s, "%Y/%m/%d")


def fmt_jalali(d: JalaliDate) -> str:
    return d.strftime("%Y/%m/%d")


def iter_date_batches(from_date: str, to_date: str, max_days_per_batch: int) -> Iterable[Tuple[str, str]]:
    """Yield inclusive Jalali [start, end] windows of at most max_days_per_batch days."""
    start = parse_jalali(from_date)
    end_all = parse_jalali(to_date)
    while start <= end_all:
        end_batch = start + timedelta(days=max_days_per_batch - 1)
        if end_batch > end_all:
            end_batch = end_all
        yield fmt_jalali(start), fmt_jalali(end_batch)
        start = end_batch + timedelta(days=1)


# ---- Core scraping logic ---------------------------------------------------

def set_date_range(page, from_date: str, to_date: str):
    """Set hidden jalali inputs directly (more reliable than interacting with UI),
    then click the Run button to generate the report.
    """
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(500)

    js = f"""
    (function() {{
        const fromInp = document.querySelector('#dnn_ctr757_Main_MainControls_rptParameters_PdpValue_0');
        const toInp   = document.querySelector('#dnn_ctr757_Main_MainControls_rptParameters_PdpValue_1');
        if (fromInp) {{
            fromInp.value = '{from_date}';
            fromInp.dispatchEvent(new Event('input', {{ bubbles: true }}));
            fromInp.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }}
        if (toInp) {{
            toInp.value = '{to_date}';
            toInp.dispatchEvent(new Event('input', {{ bubbles: true }}));
            toInp.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }}
        return !!(fromInp && toInp);
    }})();
    """
    ok = page.evaluate(js)
    if not ok:
        raise RuntimeError("Date inputs not found; page structure may have changed.")

    # Click "اجرای گزارش"
    run_sel = "#dnn_ctr757_Main_MainControls_lnkRunReport"
    page.locator(run_sel).click()
    page.wait_for_load_state("networkidle")

    # Wait for table to appear
    table_row_sel = "table.table.table-striped.table-bordered.table-hover tbody tr"
    page.wait_for_selector(table_row_sel, timeout=30000)


def scrape_one_page(page) -> List[Dict]:
    table_row_sel = "table.table.table-striped.table-bordered.table-hover tbody tr"
    rows = page.locator(table_row_sel)
    page.wait_for_selector(table_row_sel, timeout=30000)

    out: List[Dict] = []
    try:
        n = rows.count()
    except PWTimeout:
        n = 0

    for i in range(n):
        row = rows.nth(i)
        tds = row.locator("td")
        if tds.count() < 3:
            continue
        # columns: # | تاریخ | متوسط قیمت نهایی بازار (ریال بر مگاوات ساعت)
        date_jalali = tds.nth(1).inner_text().strip()
        price_txt   = tds.nth(2).inner_text().strip()
        price_val   = normalize_number(price_txt)
        if not date_jalali:
            continue
        out.append({
            "date_jalali": date_jalali,
            "average_price_irr_per_mwh": price_val,
        })
    return out


def get_first_row_date_marker(page) -> str:
    first_date_sel = "table.table.table-striped.table-bordered.table-hover tbody tr td:nth-child(2)"
    try:
        return page.locator(first_date_sel).first.inner_text().strip()
    except Exception:
        return ""


def has_next(page) -> bool:
    locator = page.locator("td.PagerOtherPageCells a.PagerHyperlinkStyle[title*='Next']")
    return locator.count() > 0


def go_next(page) -> None:
    prev_marker = get_first_row_date_marker(page)
    next_loc = page.locator("td.PagerOtherPageCells a.PagerHyperlinkStyle[title*='Next']").first
    next_loc.click()

    # Correct signature: use arg=...
    page.wait_for_function(
        """prev => {
            const el = document.querySelector(
                "table.table.table-striped.table-bordered.table-hover tbody tr td:nth-child(2)"
            );
            return el && el.textContent.trim() !== prev;
        }""",
        arg=prev_marker,
        timeout=30000,
    )
    page.wait_for_load_state("networkidle")


# ---- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IREMA average daily market price scraper")
    parser.add_argument("--from", dest="from_date", default="1392/01/01", help="Jalali start date, e.g. 1392/01/01")
    parser.add_argument("--to", dest="to_date", default="1404/06/09", help="Jalali end date, e.g. 1404/06/09")
    parser.add_argument("--out", dest="out_path", default=str(get_desktop_default()), help="Output Excel path")
    parser.add_argument("--max-days-per-batch", dest="max_days", type=int, default=1000, help="Max days per date window (controls site paging cap)")
    parser.add_argument("--headless", action="store_true", help="Run browser headless (default)")
    parser.add_argument("--headed", action="store_true", help="Run browser headed (visible)")
    args = parser.parse_args()

    headless = not args.headed  # default headless True unless --headed

    all_rows: List[Dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")

        batch_idx = 1
        for b_start, b_end in iter_date_batches(args.from_date, args.to_date, args.max_days):
            print(f"Batch {batch_idx}: {b_start} → {b_end}")
            set_date_range(page, b_start, b_end)

            # Scrape first page + iterate nexts
            page_idx = 1
            while True:
                print(f"  Scraping page {page_idx}…")
                data_page = scrape_one_page(page)
                print(f"    → {len(data_page)} rows")
                all_rows.extend(data_page)

                if has_next(page):
                    go_next(page)
                    page_idx += 1
                else:
                    break

            batch_idx += 1

        browser.close()

    if not all_rows:
        print("No data scraped. Exiting with no file.")
        return

    # Deduplicate by date (if any overlap between batches) and sort by Jalali date string
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["date_jalali"]).sort_values("date_jalali")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
