#!/usr/bin/env python3
import argparse, urllib.parse
from pathlib import Path
import requests
import pandas as pd

UA = {"User-Agent":"Mozilla/5.0"}

def get_text(url: str, timeout=30) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text

def download_csv(url: str, outdir: Path):
    fn = urllib.parse.urlparse(url).path.split("/")[-1] or "extra.csv"
    if not fn.lower().endswith(".csv"):
        fn += ".csv"
    out = outdir / fn
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"[info] extra CSV saved: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pew-url", required=True,
                    help="Pew fact sheet URL (tables will be parsed)")
    ap.add_argument("--outdir", default="./data")
    ap.add_argument("--extra-csv", action="append", default=[],
                    help="Optional direct CSV URL(s). Repeatable.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Fetching page: {args.pew_url}")
    html = get_text(args.pew_url)

    print("[info] Parsing HTML tables -> CSV")
    tables = pd.read_html(html)  # uses lxml under the hood
    if not tables:
        print("[warn] No HTML tables found on page.")
    else:
        width = len(str(len(tables)))
        for i, df in enumerate(tables, start=1):
            # Clean up obvious junk rows/cols
            df = df.dropna(how="all").dropna(axis=1, how="all")
            fn = outdir / f"pew_table_{str(i).zfill(width)}.csv"
            df.to_csv(fn, index=False)
            print(f"[info] saved table {i}: {fn.name} (rows={len(df)}, cols={len(df.columns)})")

    for url in args.extra_csv:
        try:
            download_csv(url, outdir)
        except Exception as e:
            print(f"[warn] extra CSV failed: {url} -> {e}")

    print(f"[done] Output dir: {outdir.resolve()}")

if __name__ == "__main__":
    main()
