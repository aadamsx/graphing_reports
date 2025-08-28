#!/usr/bin/env python3
# fetch_beef_data.py — BLS + ERS (no keys). NASS skipped if no key.
import argparse, os, sys, time, io
from pathlib import Path
import requests, pandas as pd

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "beef-data-fetcher/1.2 (+https://ers.usda.gov/)",
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
})

def retry_get(url: str, **kwargs) -> requests.Response:
    for i in range(5):
        r = SESSION.get(url, timeout=60, **kwargs)
        if r.status_code == 200:
            return r
        time.sleep(1.2 * (i + 1))
    r.raise_for_status()
    return r

def retry_post(url: str, json=None, **kwargs) -> requests.Response:
    for i in range(5):
        r = SESSION.post(url, json=json, timeout=60, **kwargs)
        if r.status_code == 200:
            return r
        time.sleep(1.2 * (i + 1))
    r.raise_for_status()
    return r

# -------- BLS --------
def fetch_bls(series_ids, start_year, api_key=None) -> pd.DataFrame:
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {"seriesid": series_ids, "startyear": str(start_year), "endyear": "9999"}
    if api_key:
        payload["registrationKey"] = api_key
    r = retry_post(url, json=payload)
    j = r.json()
    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API error: {j}")
    rows = []
    for s in j["Results"]["series"]:
        sid = s["seriesID"]
        for d in s["data"]:
            if not d["period"].startswith("M"):
                continue
            rows.append({
                "series_id": sid,
                "year": int(d["year"]),
                "period": d["period"],
                "period_name": d["periodName"],
                "value": float(d["value"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["month"] = df["period"].str[-2:].astype(int)
    df["date"]  = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.sort_values(["series_id","date"]).reset_index(drop=True)

# -------- ERS (no key) --------
# Source page (for reference): https://www.ers.usda.gov/data-products/meat-price-spreads
ERS_FILES = {
    # Choice beef values and spreads + all-fresh retail value
    "ers_choice_beef_values_spreads.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/beef.csv",
    # Retail prices for beef/pork/poultry cuts, eggs, dairy
    "ers_retail_cuts_prices.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/cuts.csv",
    # Summary table (handy wide table across items)
    "ers_summary_prices_spreads.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/sumtab.csv",
    # Historical monthly price spread data since 1970
    "ers_history_prices_spreads.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/history.csv",
}

def fetch_ers_to_csv(outdir: Path):
    for fname, url in ERS_FILES.items():
        try:
            r = retry_get(url, headers={"Referer": "https://www.ers.usda.gov/data-products/meat-price-spreads"})
            # Some ERS endpoints return CSV text; read and write as-is.
            # Validate by attempting to parse; still write original bytes if needed.
            try:
                df = pd.read_csv(io.StringIO(r.text))
                df.to_csv(outdir / fname, index=False)
            except Exception:
                # Fallback: write raw content
                (outdir / fname).write_bytes(r.content)
            print(f"Wrote {outdir/fname}")
        except Exception as e:
            print(f"[WARN] ERS fetch failed for {fname}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("./data"))
    ap.add_argument("--start-year", type=int, default=2000)
    args = ap.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # BLS
    try:
        bls_key = os.getenv("BLS_API_KEY") or None
        series = ["CUUR0000SEFA", "APU0000FD3101"]  # CPI beef & veal; Avg price 100% ground beef
        bls = fetch_bls(series, args.start_year, bls_key)
        bls[bls["series_id"]=="CUUR0000SEFA"].to_csv(outdir/"bls_cpi_beef_veal_monthly.csv", index=False)
        bls[bls["series_id"]=="APU0000FD3101"].to_csv(outdir/"bls_avg_price_ground_beef_monthly.csv", index=False)
        print(f"Wrote {outdir/'bls_cpi_beef_veal_monthly.csv'}")
        print(f"Wrote {outdir/'bls_avg_price_ground_beef_monthly.csv'}")
    except Exception as e:
        print(f"[WARN] BLS fetch failed: {e}", file=sys.stderr)

    # ERS
    fetch_ers_to_csv(outdir)

    # NASS skipped unless key provided later
    if not os.getenv("NASS_API_KEY"):
        print("[INFO] NASS_API_KEY not set. Skipping NASS datasets.")
    else:
        print("[INFO] NASS fetch not implemented in this keyless variant.")

    print("Done.")

if __name__ == "__main__":
    main()
