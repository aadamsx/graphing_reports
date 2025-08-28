#!/usr/bin/env python3
# analyze_beef_reports.py — builds charts + ML from your CSVs and live drought/NOAA data.
import argparse, io, os, sys, json, time, math, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- I/O helpers ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(fig, outpath: Path):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------- Load local CSVs ----------------
def load_local_datasets(datadir: Path):
    paths = {
        "bls_cpi": datadir / "bls_cpi_beef_veal_monthly.csv",
        "bls_ground": datadir / "bls_avg_price_ground_beef_monthly.csv",
        "ers_choice": datadir / "ers_choice_beef_values_spreads.csv",
        "ers_cuts": datadir / "ers_retail_cuts_prices.csv",
        "ers_sum": datadir / "ers_summary_prices_spreads.csv",
        "ers_hist": datadir / "ers_history_prices_spreads.csv",
    }
    dfs = {}
    for k, p in paths.items():
        if p.exists():
            dfs[k] = pd.read_csv(p)
        else:
            print(f"[WARN] Missing expected file: {p}", file=sys.stderr)
    return dfs

# ---------------- External data: drought + PDSI ----------------
def fetch_usdm_percent_area(start_year: int, end_date: dt.date) -> pd.DataFrame:
    url = "https://usdmdataservices.unl.edu/api/USStatistics/GetDroughtSeverityStatisticsByAreaPercent"
    params = {
        "aoi": "TOTAL",
        "startdate": f"1/1/{start_year}",
        "enddate": end_date.strftime("%m/%d/%Y"),
        "statisticsType": "1",
    }
    r = requests.get(url, params=params, timeout=60, headers={"Accept": "text/csv"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["date"] = pd.to_datetime(df["ValidStart"])
    df["pct_ge_D1"] = df[["D1","D2","D3","D4"]].sum(axis=1)
    df["pct_ge_D2"] = df[["D2","D3","D4"]].sum(axis=1)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date","pct_ge_D1","pct_ge_D2","D0","D1","D2","D3","D4","None"]]

def fetch_noaa_pdsi_monthly(start_year: int, end_date: dt.date) -> pd.DataFrame:
    this_year = end_date.year
    url = f"https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pdsi/1/0/{start_year}-{this_year}.csv"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        url = f"https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pdsi/1/13/{start_year}-{this_year}.csv"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    header_idx = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("date,"):
            header_idx = i; break
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    def parse_date(x):
        s = str(x)
        if len(s) == 6:  # YYYYMM
            return pd.to_datetime(f"{s[:4]}-{s[4:]}-01")
        return pd.to_datetime(f"{s}-01-01")
    df["date"] = df["Date"].apply(parse_date)
    df = df.rename(columns={"Value":"PDSI"})
    return df.sort_values("date")[["date","PDSI"]]

# ---------------- BLS monthly normalization ----------------
def monthly_from_bls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns and {"year","period"}.issubset(out.columns):
        out["month"] = out["period"].str[-2:].astype(int)
        out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    return out

# ---------------- Robust ERS parsers (drop-in you pasted) ----------------
def _parse_any_date_col(df: pd.DataFrame) -> pd.Series:
    cand = [c for c in df.columns if c.strip().lower() in {"date","period","time","month"}]
    if cand:
        s = pd.to_datetime(df[cand[0]], errors="coerce", infer_datetime_format=True)
        if s.notna().any():
            return s
    if {"Month","Year"}.issubset(set(df.columns)):
        try:
            m = df["Month"].astype(str).str.strip().str[:3]
            y = df["Year"].astype(int).astype(str)
            return pd.to_datetime(m + "-" + y, errors="coerce", infer_datetime_format=True)
        except Exception:
            pass
    first = df.columns[0]
    if first.lower().startswith("unnamed"):
        s = pd.to_datetime(df[first], errors="coerce", infer_datetime_format=True)
        if s.notna().any():
            return s
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if s.notna().sum() >= len(df)*0.5:
                return s
        except Exception:
            continue
    return pd.Series([pd.NaT]*len(df))

def pick_ers_all_fresh_retail(ers_choice: pd.DataFrame, ers_hist: pd.DataFrame) -> pd.DataFrame:
    for candidate in [ers_choice, ers_hist]:
        if candidate is None or candidate.empty:
            continue
        df = candidate.copy()
        df.columns = [c.strip() for c in df.columns]
        date = _parse_any_date_col(df)
        if date.isna().all():
            continue
        retail_cols = [c for c in df.columns if ("retail" in c.lower() and "beef" in c.lower() and "value" in c.lower())]
        if not retail_cols:
            retail_cols = [c for c in df.columns if ("all" in c.lower() and "fresh" in c.lower() and ("retail" in c.lower() or "value" in c.lower()))]
        if not retail_cols:
            continue
        out = pd.DataFrame({"date": date, "ers_retail_allfresh": pd.to_numeric(df[retail_cols[0]], errors="coerce")})
        out = out.dropna(subset=["date","ers_retail_allfresh"]).sort_values("date")
        if not out.empty:
            return out
    return pd.DataFrame()

def extract_choice_spreads(ers_choice: pd.DataFrame) -> pd.DataFrame:
    if ers_choice is None or ers_choice.empty:
        return pd.DataFrame()
    df = ers_choice.copy()
    df.columns = [c.strip() for c in df.columns]
    date = _parse_any_date_col(df)
    df["date"] = date
    df = df.dropna(subset=["date"])
    keep = [c for c in df.columns if c=="date" or any(k in c.lower() for k in
            ["retail value","wholesale value","farm value","spread"])]
    out = df[keep].copy()
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

# ---------------- Features ----------------
def add_time_features(df: pd.DataFrame, date_col="date"):
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["moy_sin"] = np.sin(2*np.pi*out["month"]/12)
    out["moy_cos"] = np.cos(2*np.pi*out["month"]/12)
    return out

def add_lags(df: pd.DataFrame, cols, lags=(1,3,6,12)):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            for L in lags:
                out[f"{col}_lag{L}"] = out[col].shift(L)
    return out

def rolling_feats(df: pd.DataFrame, col, windows=(3,6,12)):
    out = df.copy()
    if col in out.columns:
        for w in windows:
            out[f"{col}_ma{w}"] = out[col].rolling(w).mean()
    return out

# ---------------- Modeling ----------------
def run_ols(endog: pd.Series, exog: pd.DataFrame, out_txt: Path):
    import statsmodels.api as sm
    X = sm.add_constant(exog.dropna())
    y = endog.loc[X.index]
    model = sm.OLS(y, X).fit()
    out_txt.write_text(model.summary().as_text())
    return model

def run_sarimax(series: pd.Series, exog: pd.DataFrame, steps: int, out_png: Path):
    import statsmodels.api as sm
    s = series.dropna()
    X = None if exog is None else exog.reindex(s.index).fillna(method="ffill")
    order = (1,1,1); seasonal_order = (1,0,1,12)
    model = sm.tsa.statespace.SARIMAX(s, order=order, seasonal_order=seasonal_order, exog=X,
                                      enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    if X is not None:
        future_exog = X.iloc[-1:].repeat(steps)
    else:
        future_exog = None
    fc = res.get_forecast(steps=steps, exog=future_exog)
    pred = fc.predicted_mean; ci = fc.conf_int()
    fig, ax = plt.subplots(figsize=(10,5))
    s.tail(120).plot(ax=ax, label="Actual")
    pred.index = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    ax.plot(pred.index, pred.values, label="Forecast")
    ax.fill_between(pred.index, ci.iloc[:,0].values, ci.iloc[:,1].values, alpha=0.2)
    ax.set_title("SARIMAX forecast: retail beef")
    ax.set_ylabel("$/lb or index")
    ax.legend()
    savefig(fig, out_png)
    return res

def run_xgb_regression(train_X, train_y, test_X, test_y, out_png: Path):
    try:
        from xgboost import XGBRegressor, plot_importance
    except Exception:
        print("[INFO] xgboost not installed. Skipping advanced ML.", file=sys.stderr)
        return None
    model = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    fig, ax = plt.subplots(figsize=(8,10))
    plot_importance(model, ax=ax, max_num_features=25, importance_type="gain")
    ax.set_title("XGBoost feature importance")
    savefig(fig, out_png)
    mae = float(np.mean(np.abs(preds - test_y)))
    rmse = float(np.sqrt(np.mean((preds - test_y)**2)))
    return {"mae": mae, "rmse": rmse}

# ---------------- Plotting ----------------
def plot_price_trends(bls_ground, ers_retail, out_png: Path):
    fig, ax = plt.subplots(figsize=(10,5))
    if not ers_retail.empty:
        ax.plot(ers_retail["date"], ers_retail["ers_retail_allfresh"], label="ERS all-fresh retail beef ($/lb)")
    if not bls_ground.empty:
        ax.plot(bls_ground["date"], bls_ground["value"], label="BLS avg price 100% ground beef ($/lb)")
    ax.set_title("Retail beef price trends")
    ax.set_ylabel("$/lb")
    ax.legend()
    savefig(fig, out_png)

def plot_spreads(spreads_df, out_png: Path):
    if spreads_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10,5))
    for c in spreads_df.columns:
        if c == "date": 
            continue
        if any(k in c.lower() for k in ["spread","retail value","wholesale value","farm value"]):
            ax.plot(spreads_df["date"], spreads_df[c], label=c)
    ax.set_title("Beef values and spreads (ERS)")
    ax.legend()
    savefig(fig, out_png)

def plot_drought_overlay(price_df, drought_df, out_png: Path):
    if price_df.empty or drought_df.empty:
        return
    d = drought_df.copy()
    d["ym"] = d["date"].dt.to_period("M").dt.to_timestamp()
    m = d.groupby("ym")[["pct_ge_D1","pct_ge_D2"]].mean().reset_index().rename(columns={"ym":"date"})
    merged = pd.merge(price_df[["date","ers_retail_allfresh"]], m, on="date", how="inner")
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(merged["date"], merged["ers_retail_allfresh"], label="Retail beef ($/lb)")
    ax1.set_ylabel("$/lb")
    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["pct_ge_D1"], label="% area ≥ D1", linestyle="--")
    ax2.plot(merged["date"], merged["pct_ge_D2"], label="% area ≥ D2", linestyle=":")
    ax2.set_ylabel("% of U.S. area")
    ax1.set_title("Beef price vs U.S. drought coverage")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    savefig(fig, out_png)

def plot_corr_heatmap(df, cols, out_png: Path):
    M = df[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(M.values, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right"); ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix")
    savefig(fig, out_png)

# ---------------- Pipeline ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", type=Path, default=Path("./data"))
    ap.add_argument("--outdir", type=Path, default=Path("./reports"))
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--forecast-steps", type=int, default=12)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    local = load_local_datasets(args.datadir)
    bls_cpi = monthly_from_bls(local.get("bls_cpi", pd.DataFrame()))
    bls_ground = monthly_from_bls(local.get("bls_ground", pd.DataFrame()))
    ers_choice = local.get("ers_choice", pd.DataFrame())
    ers_hist   = local.get("ers_hist", pd.DataFrame())

    ers_retail = pick_ers_all_fresh_retail(ers_choice, ers_hist)
    spreads_df = extract_choice_spreads(ers_choice)

    today = dt.date.today()
    try:
        usdm = fetch_usdm_percent_area(args.start_year, today)
    except Exception as e:
        print(f"[WARN] USDM fetch failed: {e}", file=sys.stderr); usdm = pd.DataFrame()
    try:
        pdsi = fetch_noaa_pdsi_monthly(args.start_year, today)
    except Exception as e:
        print(f"[WARN] NOAA PDSI fetch failed: {e}", file=sys.stderr); pdsi = pd.DataFrame()

    if not ers_retail.empty or not bls_ground.empty:
        plot_price_trends(bls_ground, ers_retail, args.outdir / "1_price_trends.png")
    if not spreads_df.empty:
        plot_spreads(spreads_df, args.outdir / "2_values_spreads.png")
    if not ers_retail.empty and not usdm.empty:
        plot_drought_overlay(ers_retail, usdm, args.outdir / "3_drought_overlay.png")

    corr_df = None
    if not ers_retail.empty:
        corr_df = ers_retail[["date","ers_retail_allfresh"]].copy()
        if not bls_ground.empty:
            corr_df = corr_df.merge(
                bls_ground[["date","value"]].rename(columns={"value":"bls_ground_price"}),
                on="date", how="left"
            )
        if not pdsi.empty:
            corr_df = corr_df.merge(pdsi, on="date", how="left")
        if not usdm.empty:
            um = usdm.copy()
            um["date"] = um["date"].dt.to_period("M").dt.to_timestamp()
            um = um.groupby("date")[["pct_ge_D1","pct_ge_D2"]].mean().reset_index()
            corr_df = corr_df.merge(um, on="date", how="left")
        corr_cols = [c for c in ["ers_retail_allfresh","bls_ground_price","PDSI","pct_ge_D1","pct_ge_D2"] if c in corr_df.columns]
        if len(corr_cols) >= 2:
            plot_corr_heatmap(corr_df, corr_cols, args.outdir / "4_correlations.png")

    # Baseline ML: OLS
    if corr_df is not None and "ers_retail_allfresh" in corr_df.columns:
        df = corr_df.copy()
        df = add_time_features(df, "date")
        lag_cols = [c for c in ["pct_ge_D1","PDSI","bls_ground_price"] if c in df.columns]
        df = add_lags(df, lag_cols + ["ers_retail_allfresh"], lags=(1,3,6,12))
        for c in ["pct_ge_D1","PDSI","bls_ground_price"]:
            if c in df.columns:
                df = rolling_feats(df, c, windows=(3,6,12))
        y = df["ers_retail_allfresh"]
        X_cols = [c for c in df.columns if any(k in c for k in ["pct_ge_D1","PDSI","bls_ground_price","moy_sin","moy_cos"]) and ("lag" in c or "ma" in c or "moy_" in c)]
        X = df[X_cols]
        valid = X.dropna().index.intersection(y.dropna().index)
        if len(valid) > 24:
            ols_path = args.outdir / "5_ols_summary.txt"
            run_ols(y.loc[valid], X.loc[valid], ols_path)
            print(f"Wrote {ols_path}")

    # SARIMAX forecast
    if corr_df is not None and "ers_retail_allfresh" in corr_df.columns:
        ts = corr_df.set_index("date")["ers_retail_allfresh"].asfreq("MS")
        exog = pd.DataFrame(index=ts.index)
        if "pct_ge_D1" in corr_df.columns:
            exog["pct_ge_D1"] = corr_df.set_index("date")["pct_ge_D1"].asfreq("MS").fillna(method="ffill")
        elif "PDSI" in corr_df.columns:
            exog["PDSI"] = corr_df.set_index("date")["PDSI"].asfreq("MS").fillna(method="ffill")
        if exog.empty: exog = None
        try:
            out_png = args.outdir / "6_sarimax_forecast.png"
            run_sarimax(ts, exog, steps=args.forecast_steps, out_png=out_png)
            print(f"Wrote {out_png}")
        except Exception as e:
            print(f"[WARN] SARIMAX failed: {e}", file=sys.stderr)

    # Advanced ML
    if corr_df is not None and "ers_retail_allfresh" in corr_df.columns:
        df = corr_df.copy()
        df = add_time_features(df, "date")
        feats = [c for c in ["PDSI","pct_ge_D1","pct_ge_D2","bls_ground_price"] if c in df.columns]
        for c in feats:
            df = add_lags(df, [c], lags=(1,3,6,12))
            df = rolling_feats(df, c, windows=(3,6,12))
        df = add_lags(df, ["ers_retail_allfresh"], lags=(1,3,6,12))
        df = rolling_feats(df, "ers_retail_allfresh", windows=(3,6,12))
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if len(df) > 200:
            y = df["ers_retail_allfresh"]
            feat_cols = [c for c in df.columns if any(k in c for k in ["PDSI","pct_ge_D","bls_ground_price","ers_retail_allfresh_lag","ers_retail_allfresh_ma","moy_"])]
            X = df[feat_cols]
            split = int(len(df)*0.8)
            train_X, test_X = X.iloc[:split], X.iloc[split:]
            train_y, test_y = y.iloc[:split], y.iloc[split:]
            xgb_png = args.outdir / "7_xgb_feature_importance.png"
            metrics = run_xgb_regression(train_X, train_y, test_X, test_y, xgb_png)
            if metrics:
                (args.outdir/"7_xgb_metrics.json").write_text(json.dumps(metrics, indent=2))
                print(f"Wrote {xgb_png}")
                print(f"Wrote {(args.outdir/'7_xgb_metrics.json')}")

    if corr_df is not None:
        corr_df.to_csv(args.outdir / "merged_monthly_panel.csv", index=False)
        print(f"Wrote {(args.outdir/'merged_monthly_panel.csv')}")

    print("Done.")

if __name__ == "__main__":
    main()
