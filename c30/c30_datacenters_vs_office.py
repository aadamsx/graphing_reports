# c30_datacenters_vs_office.py
import argparse, re, unicodedata, sys
from datetime import datetime
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode()
    s = s.lower().replace("—","-").replace("–","-")
    s = re.sub(r"[^\w\s\-\./:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

MONTHS = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
          "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,
          "sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}

def parse_month_cell(x):
    if pd.isna(x): 
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.Timestamp(x.year, x.month, 1)
    # Excel serial dates
    if isinstance(x, (int, float)) and 20000 < float(x) < 60000:
        base = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(x))
        return pd.Timestamp(base.year, base.month, 1)
    s = norm(str(x))
    s = re.sub(r"([a-z]+)[\-/ ]?(\d{2,4})[rp]?$", r"\1 \2", s)  # strip r/p flags
    m = re.match(r"^([a-z]+)[\-/ ]?(\d{2,4})$", s)
    if m and m.group(1) in MONTHS:
        mm = MONTHS[m.group(1)]; yy = int(m.group(2))
        yy = (2000 + yy) if yy <= 79 else (1900 + yy) if yy < 100 else yy  # avoid 2093
        return pd.Timestamp(yy, mm, 1)
    m = re.match(r"^(19|20)\d{2}[\-/ ](1[0-2]|0?[1-9])$", s)
    if m:
        yy = int(s[:4]); mm = int(re.findall(r"(1[0-2]|0?[1-9])$", s)[0])
        return pd.Timestamp(yy, mm, 1)
    return None


def read_any(path): return pd.read_excel(path, header=None, engine="openpyxl")

def detect_date_col(df):
    best_col, best_cnt = None, 0
    for j in range(min(40, df.shape[1])):
        cnt = df.iloc[:300, j].map(parse_month_cell).notna().sum()
        if cnt > best_cnt: best_col, best_cnt = j, cnt
    return best_col if best_cnt >= 12 else None

def build_multihdr(df, date_col):
    parsed = df.iloc[:, date_col].map(parse_month_cell)
    first_idx = parsed.first_valid_index()
    if first_idx is None: sys.exit("No month rows found.")
    start_hdr = max(0, first_idx-3)
    hdr_rows = df.iloc[start_hdr:first_idx, :].fillna("").astype(str).values
    headers = []
    for c in range(df.shape[1]):
        parts = [hdr_rows[r, c] for r in range(hdr_rows.shape[0])]
        parts = [p.strip() for p in parts if p and p.strip() and p.strip().lower() != "nan"]
        headers.append(re.sub(r"\s+"," "," ".join(parts)).strip() or f"col{c}")
    data = df.iloc[first_idx:, :].copy()
    data.columns = headers
    data["date"] = data.iloc[:, date_col].map(parse_month_cell)
    data = data[pd.notna(data["date"])]
    return data

def pick_columns(cols_norm):
    def find_any(patterns):
        for c, n in cols_norm.items():
            if any(re.search(p, n) for p in patterns): return c
        return None
    col_office = find_any([r"^office\b", r"\boffice total\b"])
    col_dc     = find_any([r"\bdata\s*cent"])
    col_gen    = find_any([r"\boffice.*general\b", r"\bgeneral\b"])
    col_fin    = find_any([r"\boffice.*financ", r"\bfinanc"])
    return col_office, col_dc, col_gen, col_fin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--start", default=None, help="clip start YYYY-MM (e.g., 2020-01)")
    ap.add_argument("--end",   default=None, help="clip end YYYY-MM")
    ap.add_argument("--out", default="c30_datacenters_vs_office.png")
    ap.add_argument("--csv", default="c30_datacenters_vs_office.csv")
    ap.add_argument("--debug", action="store_true", default=False)

    args = ap.parse_args()

    raw = read_any(args.input)
    date_col = detect_date_col(raw)
    if date_col is None: sys.exit("No month column detected.")
    data = build_multihdr(raw, date_col)

    cols_norm = {c: norm(c) for c in data.columns}
    col_office, col_dc, col_gen, col_fin = pick_columns(cols_norm)
    if col_dc is None: sys.exit("No 'Data center' column found.")

    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    dc = to_num(data[col_dc])

    if col_gen:
        other = to_num(data[col_gen]); label_other = "General office"
    elif col_office:
        base = to_num(data[col_office]); fin = to_num(data[col_fin]) if col_fin else 0
        other = base - dc - fin; label_other = "Other office (Office − Data center)"
    else:
        sys.exit("Need at least 'Office' and 'Data center' columns.")

    out = pd.DataFrame({"date": data["date"], "data_centers": dc, "other_office": other}).dropna(how="all")

    if args.start: out = out[out["date"] >= pd.to_datetime(args.start + "-01")]
    if args.end:   out = out[out["date"] <= pd.to_datetime(args.end + "-01")]
    out = out.sort_values("date")

    out.to_csv(args.csv, index=False)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(out["date"], out["other_office"], label="General office")
    ax.plot(out["date"], out["data_centers"], label="Data Centers")

    ax.set_title("Private Construction: Data Centers vs Office (SAAR, $mn)")
    ax.set_xlabel("")
    ax.set_ylabel("SAAR, $mn")

    # match the exhibit's range and twin right axis
    ax.set_ylim(0, 80000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.grid(True, linewidth=0.5, alpha=0.6)

    leg = ax.legend(
        title="Value of Private Construction Put in Place:",
        loc="upper right", frameon=True, framealpha=1.0
    )

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylabel("SAAR, $mn")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.csv}, {args.out}")


if __name__ == "__main__":
    main()
