import argparse, re, unicodedata, sys
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------- utils ----------
def fmt_money(ax): ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode()
    s = re.sub(r"[^\w\s\-]", " ", s.lower()); return re.sub(r"\s+", " ", s).strip()

MONTHS = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
          "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,
          "september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}

def parse_month_cell(x):
    if pd.isna(x): return None
    if isinstance(x, (pd.Timestamp, datetime)): return pd.Timestamp(x.year, x.month, 1)
    if isinstance(x, (int,float)) and 20000 < float(x) < 60000:
        base = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(x))
        return pd.Timestamp(base.year, base.month, 1)
    s = norm(str(x)); s = re.sub(r"([a-z]+)[\-/ ]?(\d{2,4})[rp]?$", r"\1 \2", s)
    m = re.match(r"^([a-z]+)\s+(\d{2,4})$", s)
    if m and m.group(1) in MONTHS:
        mm, yy = MONTHS[m.group(1)], int(m.group(2))
        yy = 2000+yy if yy <= 79 else (1900+yy if yy < 100 else yy)
        try: return pd.Timestamp(yy, mm, 1)
        except: return None
    return None

def detect_date_col(df):
    best, cnt = None, 0
    for j in range(min(40, df.shape[1])):
        c = df.iloc[:200, j].map(parse_month_cell).notna().sum()
        if c > cnt: best, cnt = j, c
    return best if cnt >= 12 else None

def load_exog_from_xlsx(path) -> pd.DataFrame | None:
    try: raw = pd.read_excel(path, header=None, engine="openpyxl")
    except Exception: return None
    j = detect_date_col(raw); 
    if j is None: return None
    parsed = raw.iloc[:, j].map(parse_month_cell); first = parsed.first_valid_index()
    if first is None: return None
    hdr = raw.iloc[max(0, first-3):first, :].fillna("").astype(str).values
    cols = []
    for c in range(raw.shape[1]):
        parts = [hdr[r, c].strip() for r in range(hdr.shape[0]) if hdr[r, c].strip()]
        cols.append(" ".join(parts) if parts else f"col{c}")
    data = raw.iloc[first:, :].copy(); data.columns = cols
    data["date"] = raw.iloc[first:, j].map(parse_month_cell)
    data = data[pd.notna(data["date"])].set_index("date")
    labels = {c: norm(c) for c in data.columns}
    def pick(pats):
        for c, n in labels.items():
            if any(re.search(p, n) for p in pats): return c
        return None
    col_man  = pick([r"\bmanufactur"])
    col_nonr = pick([r"\bnon\s*residential\b"])
    ex = pd.DataFrame(index=data.index)
    if col_man:  ex["mfg"] = pd.to_numeric(data[col_man], errors="coerce")
    if col_nonr: ex["nonres"] = pd.to_numeric(data[col_nonr], errors="coerce")
    return ex.asfreq("MS") if not ex.empty else None

# ---------- SARIMAX helpers ----------
def sarimax_fit_forecast(y: pd.Series, exog: pd.DataFrame | None, steps: int):
    mod = SARIMAX(y.asfreq("MS"), exog=None if exog is None else exog.reindex(y.index).ffill(),
                  order=(1,1,1), seasonal_order=(1,1,0,12),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    exf = None
    if exog is not None:
        last = exog.reindex(y.index).ffill().iloc[[-1]]
        exf = pd.concat([last]*steps).set_index(pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1),
                                                              periods=steps, freq="MS"))
    fc = res.get_forecast(steps=steps, exog=exf).predicted_mean
    return res, fc

def rolling_backtest(y: pd.Series, exog: pd.DataFrame | None, horizon=6, min_train=36):
    y = y.asfreq("MS").dropna()
    X = None if exog is None else exog.reindex(y.index).ffill()
    rows = []
    for t in range(min_train, len(y) - horizon):
        y_tr = y.iloc[:t]; X_tr = None if X is None else X.iloc[:t]
        try:
            res = SARIMAX(y_tr, exog=X_tr, order=(1,1,1), seasonal_order=(1,1,0,12),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        except Exception:
            continue
        X_f = None if X is None else X.iloc[t:t+horizon]
        pred = res.get_forecast(steps=horizon, exog=X_f).predicted_mean
        for h in range(1, horizon+1):
            rows.append({"horizon": h,
                         "err": float(y.iloc[t+h-1] - pred.iloc[h-1])})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    summ = df.groupby("horizon")["err"].agg(
        RMSE=lambda e: float(np.sqrt(np.mean(e**2))),
        MAE=lambda e: float(np.mean(np.abs(e))),
    ).reset_index()
    return summ

# ---------- simple BIC-based break finder (no ruptures) ----------
def segment_sse(y):
    mu = np.mean(y); return float(np.sum((y - mu)**2))

def best_split_bic(y, min_seg=12):
    n = len(y); 
    if n < 2*min_seg: return None
    sse0 = segment_sse(y); bic0 = n*np.log(sse0/n) + 1*np.log(n)
    best = (None, np.inf)
    for i in range(min_seg, n-min_seg+1):
        ssel = segment_sse(y[:i]); sser = segment_sse(y[i:])
        bic1 = n*np.log((ssel+sser)/n) + 2*np.log(n)
        if bic1 < best[1]: best = (i, bic1)
    if best[0] is not None and best[1] + 3 < bic0:  # small guard band
        return best[0]
    return None

def binary_segmentation_bic(series: pd.Series, max_bkps=2, min_seg=12):
    y = series.dropna().values; idx = series.dropna().index
    breaks = []
    def recurse(start, end):
        if len(breaks) >= max_bkps: return
        yseg = y[start:end]; split = best_split_bic(yseg, min_seg)
        if split is None: return
        cut = start + split; breaks.append(idx[cut-1])
        recurse(start, cut); recurse(cut, end)
    recurse(0, len(y))
    return sorted(breaks)

# ---------- script ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="c30_datacenters_vs_office.csv")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--min-train", type=int, default=36)
    ap.add_argument("--exog-xlsx", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"]).set_index("date").asfreq("MS")
    office_col = "general_office" if "general_office" in df.columns else "other_office"
    dc = df["data_centers"].dropna(); office = df[office_col].dropna()

    exog = load_exog_from_xlsx(args.exog_xlsx) if args.exog_xlsx else None
    exog_dc = None if exog is None else exog.copy().assign(office=office)
    exog_off = None if exog is None else exog.copy().assign(dc=dc)

    # backtests (univariate for simplicity)
    bt_dc  = rolling_backtest(dc, None, args.horizon, args.min_train).assign(series="dc")
    bt_off = rolling_backtest(office, None, args.horizon, args.min_train).assign(series="office")
    bt = pd.concat([bt_dc, bt_off], ignore_index=True)
    bt.to_csv("c30_backtest_summary.csv", index=False)

    # forecasts
    _, dc_f   = sarimax_fit_forecast(dc,   None, args.horizon)
    _, off_f  = sarimax_fit_forecast(office, None, args.horizon)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(office.index, office, label="Office")
    ax.plot(dc.index, dc, label="Data Centers")
    ax.plot(off_f.index, off_f, "--", label="Office (forecast)")
    ax.plot(dc_f.index, dc_f, "--", label="Data Centers (forecast)")
    proj = pd.DataFrame({"dc": pd.concat([dc, dc_f]), "office": pd.concat([office, off_f])}).dropna()
    cross = proj[proj["dc"] >= proj["office"]].index.min()
    if cross is not None:
        ax.axvline(cross, linestyle=":", linewidth=1)
        ax.text(cross, ax.get_ylim()[1]*0.95, f"Projected cross: {cross:%b %Y}", rotation=90, va="top")
    ax.set_title("Data centers vs office: history and 6-month forecast")
    ax.set_ylabel("SAAR, $mn"); ax.set_xlabel("")
    fmt_money(ax); ax.xaxis.set_major_locator(mdates.YearLocator(1)); ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(); fig.tight_layout(); fig.savefig("c30_forecast_univariate.png", dpi=160)

    # changepoints (no ruptures)
    pts = binary_segmentation_bic(office, max_bkps=2, min_seg=12)
    fig2, ax2 = plt.subplots(figsize=(9,5))
    ax2.plot(office.index, office, label="Office")
    for p in pts:
        ax2.axvline(p, linestyle="--", linewidth=1); ax2.text(p, ax2.get_ylim()[1]*0.95, f"{p:%b %Y}", rotation=90, va="top")
    ax2.set_title("Office series: structural breaks (BIC binary segmentation)")
    ax2.set_ylabel("SAAR, $mn"); ax2.set_xlabel("")
    fmt_money(ax2); ax2.xaxis.set_major_locator(mdates.YearLocator(1)); ax2.grid(True, linewidth=0.5, alpha=0.6)
    ax2.legend(); fig2.tight_layout(); fig2.savefig("c30_office_changepoints.png", dpi=160)

    print("Wrote: c30_backtest_summary.csv, c30_forecast_univariate.png, c30_office_changepoints.png")

if __name__ == "__main__":
    main()



# What it produces

# c30_backtest_summary.csv with RMSE and MAE by forecast horizon.

# c30_forecast_univariate.png with history + 6-month forecasts and a projected crossover line.

# c30_office_changepoints.png with detected structural breaks in Office.

# Inputs

# --csv → your c30_datacenters_vs_office.csv (monthly).

# Optional --exog-xlsx → Census workbook to mine extra drivers.

# --horizon (default 6), --min-train (default 36).

# Data prep

# Reads CSV. Sets monthly index (asfreq("MS")).

# Picks general_office else other_office.

# If --exog-xlsx is given:

# Finds the date column in the Excel (handles Excel serials and “May-25r”).

# Collapses multi-row headers.

# Heuristically pulls Manufacturing and Nonresidential series when present.

# Aligns exog monthly.

# Forecast model (SARIMAX)

# For each series 
# 𝑦
# 𝑡
# y
# t
# 	​

#  it fits SARIMAX(1,1,1) × (1,1,0)_{12}.

# Non-seasonal: AR(1), first difference (I=1), MA(1).

# Seasonal: AR(1) at 12 months with seasonal differencing (D=1).

# This captures trend + yearly seasonality.

# If exogenous vars are supplied, they are aligned and last row is held constant for the forecast horizon.

# Produces 
# ℎ
# h-step predictions for Data Centers and for Office.

# Rolling-origin backtest

# Walks forward from min_train to the end minus horizon.

# At each cut:

# Fit on data up to 
# 𝑡
# t.

# Forecast 
# 1..
# ℎ
# 1..h months ahead.

# Record errors 
# 𝑦
# 𝑡
# +
# ℎ
# −
# 𝑦
# ^
# 𝑡
# +
# ℎ
# y
# t+h
# 	​

# −
# y
# ^
# 	​

# t+h
# 	​

# .

# Aggregates per-horizon RMSE and MAE to c30_backtest_summary.csv.

# Use this to judge horizon quality and to compare with future exog models.

# Structural breaks (no compiled deps)

# Binary segmentation with a BIC split test.

# Tries up to two breakpoints with min segment = 12 months.

# Marks dates where mean level shifts are statistically favored over a single-mean fit.

# Plots vertical lines at detected dates.

# Plot details

# Solid lines = history. Dashed = forecast.

# Crossover date = first index where projected dc ≥ office after combining history + forecast.

# Axes formatted to $mn and yearly ticks.

# How to learn by tweaking

# Change model orders: order=(p,d,q), seasonal_order=(P,D,Q,12). Watch backtest RMSE.

# Try exogenous drivers: pass --exog-xlsx privsatime.xlsx. Inspect whether RMSE falls.

# Vary --min-train and --horizon to see stability.

# Tighten or loosen the break detector: change max_bkps or min_seg, or the BIC guard band + 3.

# Export residuals and run ACF/PACF to understand remaining structure.

# That’s the whole pipeline: clean monthly data → SARIMAX forecasts → rolling evaluation → simple breakpoint dating → plots for communication.