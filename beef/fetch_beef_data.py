#!/usr/bin/env python3
# fetch_beef_data.py — BLS + ERS + drought + disasters + shipping proxy (no keys)
import argparse, os, sys, time, io, json
from pathlib import Path
import requests, pandas as pd

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "beef-data-fetcher/1.3",
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

# ---------------- BLS ----------------
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

# ---------------- ERS (no key) ----------------
ERS_FILES = {
    "ers_choice_beef_values_spreads.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/beef.csv",
    "ers_retail_cuts_prices.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/cuts.csv",
    "ers_summary_prices_spreads.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/sumtab.csv",
    "ers_history_prices_spreads.csv":
        "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/110737/history.csv",
}

def fetch_ers_to_csv(outdir: Path):
    for fname, url in ERS_FILES.items():
        try:
            r = retry_get(url, headers={"Referer": "https://www.ers.usda.gov/data-products/meat-price-spreads"})
            try:
                df = pd.read_csv(io.StringIO(r.text))
                df.to_csv(outdir / fname, index=False)
            except Exception:
                (outdir / fname).write_bytes(r.content)
            print(f"Wrote {outdir/fname}")
        except Exception as e:
            print(f"[WARN] ERS fetch failed for {fname}: {e}", file=sys.stderr)

# ---------------- Drought overlays ----------------
def fetch_usdm_percent_area(outdir: Path, start_year: int):
    url = "https://usdmdataservices.unl.edu/api/USStatistics/GetDroughtSeverityStatisticsByAreaPercent"
    params = {
        "aoi": "TOTAL",
        "startdate": f"1/1/{start_year}",
        "enddate": pd.Timestamp.today().strftime("%m/%d/%Y"),
        "statisticsType": "1",
    }
    r = retry_get(url, params=params, headers={"Accept":"text/csv"})
    df = pd.read_csv(io.StringIO(r.text))
    # Add monthly keys
    df["date"] = pd.to_datetime(df["ValidStart"])
    df["pct_ge_D1"] = df[["D1","D2","D3","D4"]].sum(axis=1)
    df["pct_ge_D2"] = df[["D2","D3","D4"]].sum(axis=1)
    df.to_csv(outdir / "usdm_drought_percent_area.csv", index=False)
    print(f"Wrote {outdir/'usdm_drought_percent_area.csv'}")

def fetch_noaa_pdsi_monthly(outdir: Path, start_year: int):
    this_year = pd.Timestamp.today().year
    url = f"https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pdsi/1/0/{start_year}-{this_year}.csv"
    r = retry_get(url)
    # Climate-at-a-Glance CSVs have a preamble; trim to the header
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    header_idx = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("date,"):
            header_idx = i; break
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    # Parse YYYYMM or YYYY
    def parse_date(x):
        s = str(x)
        return pd.to_datetime(f"{s[:4]}-{s[4:] or '01'}-01") if len(s)==6 else pd.to_datetime(f"{s}-01-01")
    df["date"] = df["Date"].apply(parse_date)
    df = df.rename(columns={"Value":"PDSI"})[["date","PDSI"]].sort_values("date")
    df.to_csv(outdir / "noaa_pdsi_conus_monthly.csv", index=False)
    print(f"Wrote {outdir/'noaa_pdsi_conus_monthly.csv'}")

# ---------------- FEMA disasters (no key) ----------------
def fetch_fema_disasters_monthly(outdir: Path, start_year: int):
    """
    OpenFEMA Disaster Declarations Summaries v2.
    We pull minimal fields and paginate via 'links'.
    """
    base = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
    # Filter by date range
    params = {
        "$select": "declarationDate,incidentType,state",
        "$filter": f"declarationDate ge '{start_year}-01-01'",
        "$top": "2000"
    }
    all_rows = []
    url = base
    while True:
        r = retry_get(url, params=params)
        j = r.json()
        rows = j.get("DisasterDeclarationsSummaries", [])
        all_rows.extend(rows)
        links = {l.get("rel"): l.get("href") for l in j.get("links", [])}
        if "next" in links:
            url = links["next"]; params = None  # next already contains the query
        else:
            break
        # avoid hammering
        time.sleep(0.25)
    if not all_rows:
        print("[WARN] FEMA returned no rows"); return
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["declarationDate"]).dt.to_period("M").dt.to_timestamp()
    # Count all incidents, plus key types
    grp = df.groupby(["date","incidentType"]).size().reset_index(name="count")
    pivot = grp.pivot(index="date", columns="incidentType", values="count").fillna(0).sort_index()
    pivot["AllIncidents"] = pivot.sum(axis=1)
    pivot.reset_index().to_csv(outdir / "fema_disasters_monthly.csv", index=False)
    print(f"Wrote {outdir/'fema_disasters_monthly.csv'}")

# ---------------- Shipping proxy: Baltic Dry Index (no key) ----------------
def fetch_bdi(outdir: Path):
    """
    Baltic Dry Index from FRED (primary) with Stooq fallback.
    Writes bdi_daily.csv and bdi_monthly.csv.
    Set SKIP_BDI=1 to skip quietly.
    """
    if os.getenv("SKIP_BDI") == "1":
        return

    import io as _io
    from urllib.parse import urlparse, urlunparse

    def _normalize(u: str) -> str:
        # Guard against 'https://https/...' or similar accidents
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            p = p._replace(scheme="https")
        if p.netloc == "https":  # e.g., 'https://https/fred...'
            # Move first path segment into netloc
            parts = [s for s in p.path.split("/") if s]
            if parts:
                p = p._replace(netloc=parts[0], path="/" + "/".join(parts[1:]))
            else:
                p = p._replace(netloc="fred.stlouisfed.org")
        return urlunparse(p)

    def _write(df: pd.DataFrame):
        df = df.dropna().sort_values("date")
        (outdir / "bdi_daily.csv").write_text(df.to_csv(index=False))
        m = df.set_index("date").resample("MS")["BDI_Close"].mean().reset_index()
        (outdir / "bdi_monthly.csv").write_text(m.to_csv(index=False))
        print(f"Wrote {outdir/'bdi_daily.csv'}")
        print(f"Wrote {outdir/'bdi_monthly.csv'}")

    # 1) FRED daily CSV (official)
    fred = "https://fred.stlouisfed.org/series/BDIY/downloaddata/BDIY.csv"
    try:
        r = retry_get(_normalize(fred), headers={"Accept": "text/csv"})
        df = pd.read_csv(_io.StringIO(r.text))
        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        val_col  = "BDIY" if "BDIY" in df.columns else df.columns[-1]
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["BDI_Close"] = pd.to_numeric(df[val_col], errors="coerce")
        _write(df[["date","BDI_Close"]])
        return
    except Exception as e:
        last_err = e  # fall through

    # 2) Stooq fallback (no key)
    for u in [
        "https://stooq.com/q/d/l/?s=bdi&i=d",
        "https://stooq.pl/q/d/l/?s=bdi&i=d",
        "http://stooq.com/q/d/l/?s=bdi&i=d",
    ]:
        try:
            r = retry_get(_normalize(u), headers={"Accept": "text/csv"})
            txt = r.text.strip()
            if "<html" in txt.lower():
                continue
            df = pd.read_csv(_io.StringIO(txt))
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get("date", list(df.columns)[0])
            # pick a numeric close-like column
            for cand in ("close","zamkniecie","last","c","adjclose"):
                if cand in cols:
                    close_col = cols[cand]
                    break
            else:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    continue
                close_col = num_cols[-1]
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.rename(columns={close_col: "BDI_Close"})[["date","BDI_Close"]]
            _write(df)
            return
        except Exception:
            continue

    print(f"[WARN] BDI fetch failed after fallbacks: {last_err}", file=sys.stderr)





# ---------------- Main ----------------
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
        series = ["CUUR0000SEFA", "APU0000FD3101"]
        bls = fetch_bls(series, args.start_year, bls_key)
        bls[bls["series_id"]=="CUUR0000SEFA"].to_csv(outdir/"bls_cpi_beef_veal_monthly.csv", index=False)
        bls[bls["series_id"]=="APU0000FD3101"].to_csv(outdir/"bls_avg_price_ground_beef_monthly.csv", index=False)
        print(f"Wrote {outdir/'bls_cpi_beef_veal_monthly.csv'}")
        print(f"Wrote {outdir/'bls_avg_price_ground_beef_monthly.csv'}")
    except Exception as e:
        print(f"[WARN] BLS fetch failed: {e}", file=sys.stderr)

    # ERS
    fetch_ers_to_csv(outdir)

    # Overlays: drought, climate, disasters, shipping
    try:
        fetch_usdm_percent_area(outdir, args.start_year)
    except Exception as e:
        print(f"[WARN] USDM fetch failed: {e}", file=sys.stderr)
    try:
        fetch_noaa_pdsi_monthly(outdir, args.start_year)
    except Exception as e:
        print(f"[WARN] NOAA PDSI fetch failed: {e}", file=sys.stderr)
    try:
        fetch_fema_disasters_monthly(outdir, args.start_year)
    except Exception as e:
        print(f"[WARN] FEMA fetch failed: {e}", file=sys.stderr)
    try:
        fetch_bdi(outdir)
    except Exception as e:
        print(f"[WARN] BDI fetch failed: {e}", file=sys.stderr)

    # Optional NASS notice
    if os.getenv("NASS_API_KEY"):
        pass
    elif os.getenv("QUIET_NASS", "0") != "1":
        print("[INFO] NASS_API_KEY not set. Skipping NASS datasets.")

    print("Done.")

if __name__ == "__main__":
    main()
