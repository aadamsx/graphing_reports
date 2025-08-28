#!/usr/bin/env python3
import pandas as pd, glob, os, re

def has_gender_cols(cols):
    s = " ".join([str(c) for c in cols]).lower()
    return any(w in s for w in ["men","male","women","female","gender","sex"])

for p in sorted(glob.glob("./data/*.csv")):
    try:
        df = pd.read_csv(p, header=0)
    except Exception:
        # try multirow header collapse
        try:
            df = pd.read_csv(p, header=[0,1])
            df.columns = [" ".join([str(x) for x in col]).strip() for col in df.columns]
        except Exception:
            continue
    cols = df.columns.tolist()
    flag = " [gender-ish]" if has_gender_cols(cols) else ""
    print(f"\n{os.path.basename(p)} -> {len(df)} rows, {len(cols)} cols{flag}")
    print("Columns:", cols[:10], ("...+%d more" % (len(cols)-10) if len(cols)>10 else ""))
    print(df.head(3).to_string(index=False))
