# assets_fetch_fred.py  (robust header/BOM handling)
import os, io, argparse, requests, pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
DEFAULT_SIDS = ["IMPCH","IMPMX","IMP5520","IMP5490","IMP5570","IMP5600","IMP5650"]

def _normalize_cols(cols):
    out = []
    for c in cols:
        c = str(c).strip().lstrip("\ufeff")  # strip BOM and whitespace
        out.append(c)
    return out

def fetch_fred_csv(sid: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{sid}.csv")

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv, */*;q=0.1",
        "Referer": "https://fred.stlouisfed.org/"
    })
    r = sess.get(FRED_URL.format(sid=sid), params={"cosd": "2017-01-01"}, timeout=60)
    r.raise_for_status()
    text = r.text

    # Quick sanity check: should look like a CSV header
    if "," not in text.splitlines()[0]:
        raise SystemExit(f"FRED returned non-CSV for {sid}. First 200 chars: {text[:200]!r}")

    # Parse without date inference first, then normalize headers
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = _normalize_cols(df.columns)

    # Accept DATE, Observation Date variants, or fallback to first column
    date_col = None
    for cand in ["DATE", "Date", "date", "observation_date", "Observation Date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]  # fallback: first column is the date

    # Choose the value column: prefer the series id if present, else the second column
    val_col = sid if sid in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    if val_col is None:
        raise SystemExit(f"CSV for {sid} has no value column. Columns={list(df.columns)}")

    # Coerce types and basic cleanup
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Save the original CSV bytes (as text) for reproducibility
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(path)

def build_arg_map(paths: dict[str,str]) -> str:
    return ",".join(f"{sid}={paths[sid]}" for sid in paths)

def main():
    ap = argparse.ArgumentParser(description="Download local FRED CSV assets and print --fred-local arg.")
    ap.add_argument("--out", default="assets", help="output directory for CSVs")
    ap.add_argument("--sids", default=",".join(DEFAULT_SIDS),
                    help="comma-separated FRED series IDs (default: 7 partner-import series)")
    args = ap.parse_args()

    sids = [s.strip() for s in args.sids.split(",") if s.strip()]
    paths = {}
    for sid in sids:
        try:
            p = fetch_fred_csv(sid, args.out)
            paths[sid] = p
            print(f"saved: {sid} -> {p}")
        except Exception as e:
            print(f"error: {sid}: {e}")

    if paths:
        arg = build_arg_map(paths)
        print("\n--fred-local argument value:")
        print(arg)
        print("\nRun example:")
        print(
            "python transshipment_reports.py "
            "--wro withhold-release-orders-findings-fy25-jun.csv "
            "--deminimis cbp_deminimis.csv "
            "--start 2017-01 --outdir out_2017 "
            f"--fred-local '{arg}'"
        )

if __name__ == "__main__":
    main()
