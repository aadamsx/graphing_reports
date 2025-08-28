#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()

def pct(x):
    if pd.isna(x): return None
    if isinstance(x,(int,float)): return float(x)
    s=str(x).strip().replace(",","")
    m=re.match(r"^(-?\d+(\.\d+)?)\s*%?$", s)
    return float(m.group(1)) if m else None

def load_csv(p):
    return pd.read_csv(p)

def generic_rep_dem(series, outdir, start_year):
    if series.empty: return
    series=series[(series["Year"]>=start_year)&(series["Year"]<=2100)]
    if series.empty: return
    plt.figure()
    plt.plot(series["Year"], series["Rep"], marker="o", label="Rep/Lean Rep")
    plt.plot(series["Year"], series["Dem"], marker="o", label="Dem/Lean Dem")
    plt.xlabel("Year"); plt.ylabel("% of adults"); plt.title(f"Party Identification Trends ({start_year}→)")
    plt.legend(); savefig(outdir/"party_id_trend_generic.png")
    gap=series.copy(); gap["RepMinusDem"]=gap["Rep"]-gap["Dem"]
    plt.figure()
    plt.plot(gap["Year"], gap["RepMinusDem"], marker="o"); plt.axhline(0, linestyle="--")
    plt.xlabel("Year"); plt.ylabel("Rep − Dem (pp)"); plt.title(f"Party ID Balance ({start_year}→)")
    savefig(outdir/"party_id_balance_generic.png")
    y0,y1=int(series["Year"].iloc[0]), int(series["Year"].iloc[-1])
    rep0,dem0=round(series["Rep"].iloc[0]), round(series["Dem"].iloc[0])
    rep1,dem1=round(series["Rep"].iloc[-1]), round(series["Dem"].iloc[-1])
    (outdir/"x_thread_stats.md").write_text(
        "\n".join([
            f"**Stats ({y0}→{y1})**",
            f"- Rep/Lean Rep: {rep0}% → {rep1}%",
            f"- Dem/Lean Dem: {dem0}% → {dem1}%",
            f"- Balance (Rep−Dem): {rep0-dem0} → {rep1-dem1} pts",
        ]), encoding="utf-8")

def extract_rep_dem(df):
    cols={c:c for c in df.columns}
    # normalize headers
    ren={}
    for c in cols:
        lc=c.lower().strip()
        if lc=="year": ren[c]="Year"
        elif lc.startswith("rep"): ren[c]="Rep"
        elif lc.startswith("gop"): ren[c]="Rep"
        elif lc.startswith("dem"): ren[c]="Dem"
    df=df.rename(columns=ren)
    need={"Year","Rep","Dem"}
    if not need.issubset(df.columns): return None
    out=df[["Year","Rep","Dem"]].copy()
    out["Year"]=pd.to_numeric(out["Year"], errors="coerce")
    out["Rep"]=out["Rep"].map(pct)
    out["Dem"]=out["Dem"].map(pct)
    return out.dropna(subset=["Year","Rep","Dem"])

def men_women_from_files(men_file, women_file, metric, start_year, outdir):
    if not (men_file and women_file): return
    men=load_csv(men_file); wom=load_csv(women_file)
    m=extract_rep_dem(men); w=extract_rep_dem(wom)
    if m is None or w is None or m.empty or w.empty: return
    key="Rep" if metric.lower().startswith("r") else "Dem"
    m=m[["Year",key]].rename(columns={key:"Men"})
    w=w[["Year",key]].rename(columns={key:"Women"})
    mw=m.merge(w, on="Year", how="inner")
    mw=mw[(mw["Year"]>=start_year)&(mw["Year"]<=2100)].sort_values("Year")
    if mw.empty: return
    # trend
    plt.figure()
    plt.plot(mw["Year"], mw["Men"], marker="o", label=f"Men – {key}")
    plt.plot(mw["Year"], mw["Women"], marker="o", label=f"Women – {key}")
    plt.xlabel("Year"); plt.ylabel("%"); plt.title(f"Men vs Women ({key}) ({start_year}→)")
    plt.legend(); savefig(outdir/"men_vs_women_trend.png")
    # gap
    plt.figure()
    plt.plot(mw["Year"], mw["Men"]-mw["Women"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Year"); plt.ylabel("Men − Women (pp)")
    plt.title(f"Gender Gap in {key} ({start_year}→)")
    savefig(outdir/"gender_gap_trend.png")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--indir", default="./data")
    ap.add_argument("--outdir", default="./reports")
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--young-men", default="./data/young_men_points.csv")
    ap.add_argument("--men-breakout", default="./data/men_breakout.csv")
    ap.add_argument("--men-file", default=None, help="CSV assumed to be MEN (e.g., pew_table_01.csv)")
    ap.add_argument("--women-file", default=None, help="CSV assumed to be WOMEN (e.g., pew_table_02.csv)")
    ap.add_argument("--metric", default="rep", choices=["rep","dem"], help="Which share to chart for men vs women")
    args=ap.parse_args()

    indir, outdir=Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # generic Rep/Dem from any table (use pew_table_01 as baseline overall men table if desired)
    # pick one with Year/Rep/ Dem; if many, just use first
    any_csv = next(iter(list(indir.glob("*.csv"))), None)
    if any_csv:
        # try table_01 as it looks like a clean Year/Rep/Dem
        base = indir/"pew_table_01.csv"
        use = base if base.exists() else any_csv
        df=pd.read_csv(use)
        rd=extract_rep_dem(df)
        if rd is not None and not rd.empty:
            generic_rep_dem(rd, outdir, args.start_year)

    # men vs women from chosen files
    men_women_from_files(
        men_file=str(indir/args.men_file) if args.men_file else None,
        women_file=str(indir/args.women_file) if args.women_file else None,
        metric=args.metric,
        start_year=args.start_year,
        outdir=outdir
    )

    # optional charts
    try:
        ym=pd.read_csv(args.young_men)
        if {"Year","MenUnder30_Dem","MenUnder30_Rep"}.issubset(ym.columns):
            ym=ym.sort_values("Year")
            plt.figure()
            plt.plot(ym["Year"], ym["MenUnder30_Dem"], marker="o", label="Men <30 – Dem")
            plt.plot(ym["Year"], ym["MenUnder30_Rep"], marker="o", label="Men <30 – Rep")
            plt.xlabel("Year"); plt.ylabel("%"); plt.title("Young Men: Party Preference/ID"); plt.legend()
            savefig(outdir/"young_men_shift_points.png")
    except Exception: pass

    try:
        mb=pd.read_csv(args.men_breakout)
        if {"Subgroup","Rep","Dem"}.issubset(mb.columns):
            mb=mb.sort_values("Subgroup")
            plt.figure()
            plt.bar(mb["Subgroup"], mb["Rep"], label="Rep/Lean Rep")
            plt.bar(mb["Subgroup"], mb["Dem"], alpha=0.5, label="Dem/Lean Dem")
            plt.xticks(rotation=25, ha="right"); plt.ylabel("%"); plt.title("Men by Subgroup"); plt.legend()
            savefig(outdir/"men_by_subgroup.png")
    except Exception: pass

    print(f"[done] Reports in: {outdir.resolve()}")

if __name__=="__main__":
    main()
