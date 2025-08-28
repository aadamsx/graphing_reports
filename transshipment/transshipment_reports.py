import io, re, os, argparse, glob, textwrap, pandas as pd, matplotlib.pyplot as plt, requests
from matplotlib.ticker import FuncFormatter
from datetime import datetime

# ---------------- helpers ----------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SERIES = {
    "CHN":"IMPCH", "MEX":"IMPMX",
    "VNM":"IMP5520", "THA":"IMP5490", "MYS":"IMP5570", "IDN":"IMP5600", "PHL":"IMP5650",
}

def moneyfmt(ax): ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v):,}"))

def fred_series(sid: str, local_map: dict | None = None) -> pd.Series:
    # Local override (use a pre-downloaded FRED CSV)
    if local_map and sid in local_map and os.path.exists(local_map[sid]):
        df = pd.read_csv(local_map[sid])
        # FRED files usually have columns: DATE, <SID>
        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        val_col  = [c for c in df.columns if c != date_col][0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        s = pd.to_numeric(df[val_col], errors="coerce")
        s.index = df[date_col]
        return s.asfreq("MS").dropna()

    # Online fetch with proper headers and a start-date hint
    url = FRED_CSV.format(sid=sid)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv, */*;q=0.1",
        "Referer": "https://fred.stlouisfed.org/"
    })
    # add a start date to reduce redirects
    r = session.get(url, params={"cosd": "2017-01-01"}, timeout=60)
    r.raise_for_status()
    text = r.text
    # If we didn't get a CSV header, fail fast with a helpful message
    if "DATE" not in text.splitlines()[0].upper():
        raise RuntimeError(
            f"FRED fetch for {sid} did not return CSV. "
            f"HTTP {r.status_code}. First 200 chars: {text[:200]!r}. "
            f"Workaround: download CSV in browser and pass via --fred-local."
        )
    df = pd.read_csv(io.StringIO(text), parse_dates=["DATE"])
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = df["DATE"]
    return s.asfreq("MS").dropna()

def parse_num(s):
    if s is None: return None
    x = str(s).strip().lower().replace(",", "")
    m = re.match(r"^([\d\.]+)\s*([kmb])?$", x)
    if not m: return None
    val = float(m.group(1)); unit = (m.group(2) or "")
    mul = {"k":1_000, "m":1_000_000, "b":1_000_000_000}.get(unit, 1)
    return int(val*mul)

# ---------------- de minimis ----------------
def load_deminimis(path:str|None)->pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # best-effort scrape (fallback)
        url = "https://www.cbp.gov/trade/basic-import-export/e-commerce"
        html = requests.get(url, timeout=60).text
        try:
            tables = pd.read_html(html, flavor="lxml")
        except Exception:
            tables = []
        df = None
        for t in tables:
            if t.astype(str).apply(lambda c: c.str.contains("De minimis", case=False, na=False)).any().any():
                df = t; break
        if df is None: return pd.DataFrame()
    # normalize
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    fy_col = next((c for c in df.columns if "fiscal" in c and "year" in c), None)
    tot_col = next((c for c in df.columns if "total" in c and "de" in c and "min" in c), None)
    if fy_col and tot_col:
        out = df[[fy_col, tot_col]].rename(columns={fy_col:"fiscal_year", tot_col:"entries"})
        out["entries"] = out["entries"].map(parse_num)
        out = out.dropna().reset_index(drop=True)
        return out
    # fallback: try regex row extraction
    return pd.DataFrame()

# ---------------- WRO CSV ----------------
DATE_COL_PAT = re.compile(r"(date|issued|initiated|finding)", re.I)
COUNTRY_COL_PAT = re.compile(r"(country)", re.I)
TYPE_COL_PAT = re.compile(r"(type|wro|finding)", re.I)

def load_wro(path:str|None)->pd.DataFrame:
    f = path
    if not f:
        cands = sorted(glob.glob("*withhold*release*orders*findings*.csv")) or sorted(glob.glob("*wro*find*.csv"))
        f = cands[0] if cands else None
    if not f or not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f)
    # detect columns
    date_col = next((c for c in df.columns if DATE_COL_PAT.search(c)), None)
    ctry_col = next((c for c in df.columns if COUNTRY_COL_PAT.search(c)), None)
    type_col = next((c for c in df.columns if TYPE_COL_PAT.search(c)), None)
    # parse
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df["year"]  = df["date"].dt.year
    else:
        df["date"]=pd.NaT; df["month"]=pd.NaT; df["year"]=pd.NA
    if ctry_col: df["country"]=df[ctry_col].astype(str)
    if type_col: df["action_type"]=df[type_col].astype(str)
    return df

# ---------------- main build ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fred-local", default=None,
        help="Comma-separated sid=path pairs, e.g. IMPCH=assets/IMPCH.csv,IMPMX=assets/IMPMX.csv")
    ap.add_argument("--wro", default=None, help="path to withhold-release-orders-findings*.csv")
    ap.add_argument("--deminimis", default=None, help="path to cbp_deminimis.csv")
    ap.add_argument("--start", default="2017-01", help="clip start YYYY-MM for charts")
    ap.add_argument("--outdir", default="out", help="output directory")
    args = ap.parse_args()

    # Build local FRED map from --fred-local
    local_map = {}
    if args.fred_local:
        for pair in args.fred_local.split(","):
            if "=" in pair:
                sid, path = pair.split("=", 1)
                local_map[sid.strip()] = path.strip()

    # Autodetect assets/*.csv if --fred-local not supplied
    if not local_map and os.path.isdir("assets"):
        for sid in SERIES.values():
            p = os.path.join("assets", f"{sid}.csv")
            if os.path.exists(p):
                local_map[sid] = p

    os.makedirs(args.outdir, exist_ok=True)
    start = pd.to_datetime(args.start + "-01")

    # 1) FRED imports: China vs Mexico; China vs ASEAN-5
    chn   = fred_series(SERIES["CHN"], local_map)
    mex   = fred_series(SERIES["MEX"], local_map).reindex(chn.index)
    asean = sum(fred_series(SERIES[k], local_map) for k in ["VNM","THA","MYS","IDN","PHL"])
    # clip
    chn, mex = chn[chn.index >= start], mex[mex.index >= start]
    asean = asean[asean.index >= start]

    pd.DataFrame({"date": chn.index, "china": chn.values, "mexico": mex.values}) \
      .to_csv(os.path.join(args.outdir, "imports_china_mexico.csv"), index=False)
    pd.DataFrame({"date": asean.index, "china": chn.reindex(asean.index).values, "asean5": asean.values}) \
      .to_csv(os.path.join(args.outdir, "imports_china_asean5.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(chn.index, chn.values, label="China")
    ax.plot(mex.index, mex.values, label="Mexico")
    ax.set_title("U.S. Goods Imports: China vs Mexico (NSA, $mn)"); ax.set_xlabel(""); ax.set_ylabel("Millions of $")
    moneyfmt(ax); ax.grid(True, linewidth=0.5, alpha=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "chart_imports_china_mexico.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(chn.index, chn.values, label="China")
    ax.plot(asean.index, asean.values, label="ASEAN-5 (VN, TH, MY, ID, PH)")
    ax.set_title("U.S. Goods Imports: China vs ASEAN-5 Sum (NSA, $mn)"); ax.set_xlabel(""); ax.set_ylabel("Millions of $")
    moneyfmt(ax); ax.grid(True, linewidth=0.5, alpha=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "chart_imports_china_asean5.png"), dpi=160)

    # 2) De minimis bar (prefer local file)
    dm = load_deminimis(args.deminimis)
    dm_path = None
    if not dm.empty:
        dm_path = os.path.join(args.outdir, "cbp_deminimis_clean.csv")
        dm.to_csv(dm_path, index=False)
        fig, ax = plt.subplots(figsize=(9,5))
        ax.bar(dm["fiscal_year"], dm["entries"])
        ax.set_title("CBP Section 321 (De minimis) shipments by fiscal year"); ax.set_ylabel("Shipments (count)"); ax.set_xlabel("")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "chart_deminimis.png"), dpi=160)

    # 3) WRO/Findings (from downloaded CSV)
    wro = load_wro(args.wro)
    wro_year_png = wro_country_png = None
    if not wro.empty and wro["date"].notna().any():
        by_year = (wro.dropna(subset=["date"])
                      .groupby("year", dropna=True)
                      .size()
                      .reset_index(name="count"))
        by_year = by_year[by_year["year"] >= 2017]
        by_year.to_csv(os.path.join(args.outdir, "wro_counts_by_year.csv"), index=False)

        fig, ax = plt.subplots(figsize=(9,5))
        ax.bar(by_year["year"].astype(str), by_year["count"])
        ax.set_title("CBP Withhold Release Orders / Findings issued per year"); ax.set_ylabel("Count"); ax.set_xlabel("")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        wro_year_png = os.path.join(args.outdir, "chart_wro_by_year.png")
        fig.savefig(wro_year_png, dpi=160)

        if "country" in wro.columns:
            top = (wro.groupby("country").size().sort_values(ascending=False).head(12).reset_index(name="count"))
            top.to_csv(os.path.join(args.outdir, "wro_top_countries.csv"), index=False)
            fig, ax = plt.subplots(figsize=(9,5))
            ax.barh(top["country"][::-1], top["count"][::-1])
            ax.set_title("WRO/Findings by country (top 12)"); ax.set_xlabel("Count"); ax.set_ylabel("")
            ax.grid(True, axis="x", linewidth=0.5, alpha=0.6)
            fig.tight_layout()
            wro_country_png = os.path.join(args.outdir, "chart_wro_top_countries.png")
            fig.savefig(wro_country_png, dpi=160)

    # 4) Report (Markdown)
    md = []
    md.append("# Transshipment Tariffs — Open-source Indicators Report\n")
    md.append(f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
    md.append("## Imports by partner\n")
    md.append("![China vs Mexico](chart_imports_china_mexico.png)\n")
    md.append("![China vs ASEAN-5](chart_imports_china_asean5.png)\n")
    if dm_path:
        md.append("## De minimis (Section 321) shipments\n")
        md.append("![De minimis](chart_deminimis.png)\n")
    if wro_year_png:
        md.append("## Enforcement actions (WRO/Findings)\n")
        md.append("![WRO by year](chart_wro_by_year.png)\n")
    if wro_country_png:
        md.append("![Top countries](chart_wro_top_countries.png)\n")
    md.append("\n_Data sources: FRED partner import series; CBP e-commerce page for de minimis; CBP WRO/Findings CSV you downloaded._\n")
    with open(os.path.join(args.outdir, "transshipment_report.md"), "w") as f:
        f.write("\n".join(md))

    print("Wrote report to", os.path.join(args.outdir, "transshipment_report.md"))

if __name__ == "__main__":
    main()
