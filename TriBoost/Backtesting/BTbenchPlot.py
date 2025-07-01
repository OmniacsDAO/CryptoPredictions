from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --------------------------------------------------------------------------
# Config – reuse the paths from your main script
RESULTS_PATH = Path("temp/ResBTbenchBT6.pkl")   # original file
PLOT_PATH = Path("temp/PlotBTbenchBT6.png")

# --------------------------------------------------------------------------
# 1. Load the existing results dict
with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)

if not results:
    raise ValueError("Results dict is empty – nothing to trim!")

dfs = [val for val in results.values() if isinstance(val, pd.DataFrame)]
results[min(results.keys())] = pd.concat(dfs, ignore_index=True)


# ── 1.  Calculate 8-hour returns ────────────────────────────────────────────────
df = results[min(results.keys())]
df["act_ret"]  = df["future_close"] / df["close"] - 1.0
df["pred_ret"] = df["pred_close"]  / df["close"] - 1.0

# ── 2. Create the base scatter plot ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
sns.scatterplot(data=df, x="act_ret", y="pred_ret",
                s=20, alpha=0.8, linewidth=0, ax=ax)

lims = np.percentile(df[["act_ret", "pred_ret"]].values, [0.5, 99.5])
ax.plot(lims, lims, ls="--", c="gray", label="x = y")
ax.set(xlim=lims, ylim=lims,
       xlabel="Actual Returns", ylabel="Predicted Returns")
ax.legend()

# ── 3. Optional quadrant-percentage labels ───────────────────────────────
def quadrant_flags(x, y):
    if   x >= 0 and y >= 0: return "Q1"
    elif x < 0 and y >= 0:  return "Q2"
    elif x < 0 and y < 0:   return "Q3"
    else:                   return "Q4"

df["quad"] = [quadrant_flags(x, y) for x, y in zip(df.act_ret, df.pred_ret)]
quad_pct   = df["quad"].value_counts(normalize=True).mul(100).round(2)

for q, pct in quad_pct.items():
    x_pos = 0.02 if q in ["Q1", "Q2"] else -0.02
    y_pos = 0.02 if q in ["Q1", "Q4"] else -0.02
    ax.text(x_pos, y_pos, f"{pct:.2f}%", ha="center", va="center",
            fontsize=12, weight="bold", transform=ax.transData)

# ── 4. Optional correlation annotation ───────────────────────────────────
r, _ = pearsonr(df.act_ret, df.pred_ret)
ax.text(0.98, 0.02, f"$\\rho$ = {r:.3f}\n$R^2$ = {r**2:.3f}",
        ha="right", va="bottom", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# ── 5. Final layout tweaks & save AFTER all annotations ───────────────────
fig.tight_layout()
fig.savefig("actual_vs_predicted_returns.png", dpi=300, bbox_inches="tight")
