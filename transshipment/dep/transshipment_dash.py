import io, re, requests, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------- helpers ----------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

SERIES = {
    "CHN":"IMPCH", "MEX":"IMPMX",
    "VNM":"IMP5520", "THA":"IMP5490", "MYS":"IMP5570", "IDN":"IMP5600", "PHL":"IMP5650",
}

def fred_series(sid:str)->pd.Series:
    r = requests.get(FRED_CSV.format(sid=sid), timeout=60); r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), parse_dates=["DATE"])
    s = pd.to_numeric(df.iloc[:,1], errors="coerce")
    s.index = df["DATE"]
    return s.asfreq("MS").dropna()

def parse_num(s):
    if s is None: return None
    x = str(s).strip().lower().replace(",", "")
    m = re.match(r"^([\d\.]+)\s*([kmb])?$", x)
    if not m: return None
    val = float(m.group(1)); unit = m.group(2) or ""
    mul = {"k":1_000, "m":1_000_000, "b":1_000_000_000}.get(unit, 1)
    return int(val*mul)

def fetch_deminimis()->pd.DataFrame:
    url = "https://www.cbp.gov/trade/basic-import-export/e-commerce"
    html = requests.get(url, timeout=60).text
    # Try table scrape first
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        tables = []
    df = None
    for t in tables:
        # look for a FY row
        if any(t.astype(str).apply(lambda col: col.str.contains("Fiscal Year", case=False, na=False)).any()):
            df = t; break
    if df is None:
        # fallback: regex the key lines on the page snapshot
        # Find the block listing Fiscal Year and Total De minimis
        block = re.search(r"Fiscal Year.*?Total De minimis.*?(?:\n|$)", html, re.I|re.S)
        # If not found, just craft from the known label lines
        block = html
        # Pull totals row
    # Regex for the totals row numbers in order 2020..2025
    totals = re.search(r"Total De minimis\s*([0-9\.\sMBK]+)\s([0-9\.\sMBK]+)\s([0-9\.\sMBK]+)\s([0-9\.\sMBK]+)\s([0-9\.\sMBK]+)\s([0-9\.\sMBK]+)\s([0-9\.\sMBK]+)?", html, re.I)
    years  = ["2020","2021","2022","2023","2024","2025 (Oct–Apr)","2025 (May–Jun)"]
    vals = []
    if totals:
        nums = totals.groups()
        for y, n in zip(years, nums):
            vals.append({"fiscal_year": y, "entries": parse_num(n)})
    else:
        vals = []
    return pd.DataFrame(vals).dropna()

def moneyfmt(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v):,}"))

# ---------- build series ----------
chn = fred_series(SERIES["CHN"])
mex = fred_series(SERIES["MEX"])
asean = sum(fred_series(SERIES[k]) for k in ["VNM","THA","MYS","IDN","PHL"])

# ---------- save CSVs ----------
pd.DataFrame({"date": chn.index, "china": chn.values, "mexico": mex.reindex(chn.index).values}).to_csv("imports_china_mexico.csv", index=False)
pd.DataFrame({"date": asean.index, "china": chn.reindex(asean.index).values, "asean5": asean.values}).to_csv("imports_china_asean5.csv", index=False)

# ---------- plot 1: China vs Mexico ----------
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(chn.index, chn.values, label="China")
ax.plot(mex.index, mex.values, label="Mexico")
ax.set_title("U.S. Goods Imports: China vs Mexico (NSA, $mn)")
ax.set_xlabel(""); ax.set_ylabel("Millions of $")
moneyfmt(ax); ax.grid(True, linewidth=0.5, alpha=0.6); ax.legend()
fig.tight_layout(); fig.savefig("chart_imports_china_mexico.png", dpi=160)

# ---------- plot 2: China vs ASEAN-5 sum ----------
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(chn.index, chn.values, label="China")
ax.plot(asean.index, asean.values, label="ASEAN-5 (VN, TH, MY, ID, PH)")
ax.set_title("U.S. Goods Imports: China vs ASEAN-5 Sum (NSA, $mn)")
ax.set_xlabel(""); ax.set_ylabel("Millions of $")
moneyfmt(ax); ax.grid(True, linewidth=0.5, alpha=0.6); ax.legend()
fig.tight_layout(); fig.savefig("chart_imports_china_asean5.png", dpi=160)

# ---------- plot 3: De minimis (FY) ----------
dm = fetch_deminimis()
if not dm.empty:
    dm.to_csv("cbp_deminimis.csv", index=False)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(dm["fiscal_year"], dm["entries"])
    ax.set_title("CBP Section 321 (De minimis) shipments by fiscal year")
    ax.set_ylabel("Shipments (count)"); ax.set_xlabel("")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig("chart_deminimis.png", dpi=160)
    print("Wrote imports_china_mexico.csv, imports_china_asean5.csv, cbp_deminimis.csv and PNGs.")
else:
    print("Wrote imports CSVs and PNGs. De minimis table not parsed (page structure changed).")
