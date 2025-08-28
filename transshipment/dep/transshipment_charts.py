import io, re, requests, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SERIES = {
    # partners
    "CHN":"IMPCH", "MEX":"IMPMX",
    "VNM":"IMP5520", "THA":"IMP5490", "MYS":"IMP5570", "IDN":"IMP5600", "PHL":"IMP5650",
}
START = pd.Timestamp("2017-01-01")

def fred_series(sid):
    r = requests.get(FRED.format(sid=sid), timeout=60); r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.rename(columns={df.columns[1]: "value"}, inplace=True)
    return df[["DATE","value"]].replace(".", pd.NA).dropna().assign(value=lambda d: pd.to_numeric(d["value"]))

def plot_line(df, cols, title, ylab, outpng):
    fig, ax = plt.subplots(figsize=(9,5))
    for c in cols:
        ax.plot(df.index, df[c], label=c)
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel(ylab); ax.grid(True, linewidth=0.5, alpha=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(outpng, dpi=160)

# 1) China vs Mexico
chn = fred_series(SERIES["CHN"])
mex = fred_series(SERIES["MEX"])
df1 = (chn.merge(mex, on="DATE", how="outer", suffixes=("_CHN","_MEX"))
          .set_index("DATE").sort_index()).loc[START:]
plot_line(df1, ["value_CHN","value_MEX"],
          "U.S. Goods Imports: China vs Mexico (NSA, $M)", "Millions of $",
          "chart_imports_china_mexico.png")

# 2) China vs ASEAN-5 (sum)
asean_ids = ["VNM","THA","MYS","IDN","PHL"]
asean = None
for k in asean_ids:
    s = fred_series(SERIES[k]).rename(columns={"value": f"value_{k}"})
    asean = s if asean is None else asean.merge(s, on="DATE", how="outer")
asean["ASEAN5"] = asean.filter(like="value_").sum(axis=1)
df2 = (asean[["DATE","ASEAN5"]]
       .merge(chn, on="DATE", how="outer")
       .rename(columns={"value":"CHN"})
       .set_index("DATE").sort_index()).loc[START:]
plot_line(df2, ["CHN","ASEAN5"],
          "U.S. Goods Imports: China vs ASEAN-5 Sum (NSA, $M)", "Millions of $",
          "chart_imports_china_asean5.png")

# 3) De minimis (Section 321) volumes by fiscal year
def _num_to_int(x):
    if x is None: return None
    s = str(x).lower().replace(",", "").strip()
    m = re.search(r"([\d\.]+)\s*(billion|b|million|m)", s)
    if not m: 
        try: return int(float(s))
        except: return None
    val = float(m.group(1)); unit = m.group(2)
    return int(val * (1_000_000_000 if unit.startswith("b") else 1_000_000))

def fetch_deminimis():
    # Primary: CBP e-commerce page
    url1 = "https://www.cbp.gov/trade/basic-import-export/e-commerce"
    tables = pd.read_html(requests.get(url1, timeout=60).text)
    target = None
    for t in tables:
        if any("De minimis" in str(c) for c in t.columns) or t.apply(lambda r: r.astype(str).str.contains("De minimis", case=False).any(), axis=1).any():
            target = t; break
    if target is None:
        # Fallback: Trade statistics page
        url2 = "https://www.cbp.gov/newsroom/stats/trade"
        tables = pd.read_html(requests.get(url2, timeout=60).text)
        for t in tables:
            if t.shape[1] >= 3 and t.iloc[:,0].astype(str).str.contains("Section 321", case=False).any():
                target = t; break
    if target is None:
        raise RuntimeError("Could not locate de minimis table on CBP pages.")
    # Normalize wide FY table
    # Find the row with total de minimis counts
    rmask = target.apply(lambda r: r.astype(str).str.contains("Total De minimis|Section 321 BOL", case=False).any(), axis=1)
    row = target[rmask].iloc[0]
    # Build year→value mapping from header labels that look like years
    out = []
    for col in target.columns:
        if re.search(r"\b20\d{2}\b", str(col)):
            out.append({"fiscal_year": str(col), "entries": _num_to_int(row[col])})
    df = pd.DataFrame(out).dropna().sort_values("fiscal_year")
    return df

dm = fetch_deminimis()
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(dm["fiscal_year"], dm["entries"])
ax.set_title("De minimis (Section 321) entries by fiscal year"); ax.set_ylabel("Entries (count)")
ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
fig.tight_layout(); fig.savefig("chart_deminimis.png", dpi=160)

# 4) Customs duties revenue (optional)
try:
    duties = fred_series("B235RC1A027NBEA")  # annual
    duties["YEAR"] = duties["DATE"].dt.year
    d = duties.groupby("YEAR", as_index=False)["value"].first()
    d = d[d["YEAR"]>=2015]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(d["YEAR"], d["value"])
    ax.set_title("Customs duties (federal receipts), annual"); ax.set_xlabel(""); ax.set_ylabel("Billions of $")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig("chart_customs_duties.png", dpi=160)
except Exception as e:
    print("Customs duties chart skipped:", e)

print("Wrote:",
      "chart_imports_china_mexico.png,",
      "chart_imports_china_asean5.png,",
      "chart_deminimis.png,",
      "chart_customs_duties.png")
