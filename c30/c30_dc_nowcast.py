# c30_dc_nowcast.py
# Your file loads monthly series, fits SARIMAX(1,1,1)×(1,1,0)_12 to each, forecasts 6 months, appends forecasts, then draws the history and dashed forecasts.

# SARIMAX = ARIMA with seasonal terms and optional exogenous regressors. Here it models:

# one non-seasonal AR(1), I(1), MA(1)

# one seasonal AR(1) at 12-month period with seasonal differencing I_12(1)

# Crossing month = first date where DataCenters ≥ Office in the combined actual+forecast series.

# Plot uses Matplotlib. Ticks = yearly (mdates.YearLocator(1)), values formatted in $mn.

import pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.ticker import FuncFormatter

df = pd.read_csv("c30_datacenters_vs_office.csv", parse_dates=["date"]).set_index("date")
dc = df["data_centers"].dropna().asfreq("MS")
office_col = "general_office" if "general_office" in df.columns else "other_office"
office = df[office_col].dropna().asfreq("MS")

def fcst(s, steps=6):
    m = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,0,12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m.get_forecast(steps=steps).predicted_mean

h = 6
dc_f, off_f = fcst(dc, h), fcst(office, h)

dc_proj   = pd.concat([dc, dc_f])
off_proj  = pd.concat([office, off_f])
proj = pd.DataFrame({"dc": dc_proj, "office": off_proj})

cross = proj[proj["dc"] >= proj["office"]].index.min()

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(dc.index, dc, label="Data Centers")
ax.plot(office.index, office, label="Office")
ax.plot(dc_f.index, dc_f, "--", label="Data Centers (forecast)")
ax.plot(off_f.index, off_f, "--", label="Office (forecast)")
if cross is not None:
    ax.axvline(cross, linestyle=":", linewidth=1)
    ax.text(cross, ax.get_ylim()[1]*0.95, f"Projected cross: {cross:%b %Y}", rotation=90, va="top")
ax.set_title("Data centers vs office with 6-month SARIMAX forecast")
ax.set_ylabel("SAAR, $mn"); ax.set_xlabel("")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.grid(True, linewidth=0.5, alpha=0.6); ax.legend()
fig.tight_layout(); fig.savefig("c30_projection.png", dpi=160)
print("Projected cross month:", None if cross is None else cross.strftime("%Y-%m"))
