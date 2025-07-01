import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import binomtest, pearsonr

def evaluate_monthly_metrics(results, prediction_column="pred_close"):
    metrics = {
        'Month': [],
        'Directional Accuracy': [],
        'DA_Lower': [],
        'DA_Upper': [],
        'DA_pval': [],
        'Pearson r': [],
        'Pearson pval': [],
        'WRMSE (%)': [],
        'WMZTAE (%)': []
    }

    for i, df in results.items():
        df = df.copy()
        df['FuturePrice'] = df['future_close']
        df['ReferencePrice'] = df['close']
        df['Prediction'] = df[prediction_column]
        df = df.dropna()

        actual_returns = np.log(df.FuturePrice / df.ReferencePrice)
        predicted_returns = np.log(df.Prediction / df.ReferencePrice)

        # Directional Accuracy with CI
        successes = np.sum(actual_returns * predicted_returns > 0)
        invalid = np.sum(actual_returns == 0)
        n = len(actual_returns) - invalid
        da = successes / n if n > 0 else np.nan
        test_result = binomtest(successes, n, p=0.5)
        ci = test_result.proportion_ci(confidence_level=0.95)
        metrics['Directional Accuracy'].append(da)
        metrics['DA_Lower'].append(ci.low)
        metrics['DA_Upper'].append(ci.high)
        metrics['DA_pval'].append(test_result.pvalue)

        # Pearson correlation
        r, pval = pearsonr(predicted_returns, actual_returns)
        metrics['Pearson r'].append(r)
        metrics['Pearson pval'].append(pval)

        # WRMSE Improvement
        weights = np.abs(actual_returns)
        wrmse_baseline = np.sqrt(np.sum(weights * (actual_returns - 0)**2) / np.sum(weights))
        wrmse_model = np.sqrt(np.sum(weights * (actual_returns - predicted_returns)**2) / np.sum(weights))
        wrmse_improvement = (wrmse_baseline - wrmse_model) / wrmse_baseline * 100
        metrics['WRMSE (%)'].append(wrmse_improvement)

        # WMZTAE Improvement
        stdev = np.std(actual_returns)
        wmztae_baseline = np.sum(weights * np.abs(np.tanh(actual_returns / stdev))) / np.sum(weights)
        wmztae_model = np.sum(weights * np.abs(np.tanh(actual_returns / stdev) - np.tanh(predicted_returns / stdev))) / np.sum(weights)
        wmztae_improvement = (wmztae_baseline - wmztae_model) / wmztae_baseline * 100
        metrics['WMZTAE (%)'].append(wmztae_improvement)

        # Month label
        month_label = df['date'].iloc[0].strftime('%Y-%m')
        metrics['Month'].append(pd.to_datetime(month_label))

    return pd.DataFrame(metrics).sort_values("Month")

def plot_metric_timeseries(metrics_df, filename="metrics_plot.png"):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # -- Helper to auto-scale y-axis with margin and force inclusion of baselines --
    def set_y_range(ax, values, margin=0.1, include_baseline=None):
        values = np.array(values, dtype=np.float64)
        if include_baseline is not None:
            if not isinstance(include_baseline, (list, tuple, np.ndarray)):
                include_baseline = [include_baseline]
            values = np.concatenate([values, np.array(include_baseline)])
        min_val, max_val = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return
        span = max_val - min_val
        if span == 0:
            span = 0.1 * abs(min_val) if min_val != 0 else 0.1
        ax.set_ylim(min_val - margin * span, max_val + margin * span)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Monthly Evaluation Metrics", fontsize=16)

    months = metrics_df["Month"]

    # --- Plot 1: Directional Accuracy (DA) ---
    colors_da = ['red' if p > 0.05 else 'black' for p in metrics_df["DA_pval"]]
    axs[0, 0].fill_between(months, metrics_df["DA_Lower"], metrics_df["DA_Upper"], alpha=0.2)
    axs[0, 0].scatter(months, metrics_df["Directional Accuracy"], color=colors_da)
    axs[0, 0].plot(months, metrics_df["Directional Accuracy"], color='gray')
    axs[0, 0].axhline(0.52, color='red', linestyle='--', label='Min threshold (0.52)')
    axs[0, 0].axhline(0.55, color='black', linestyle='--', label='Target threshold (0.55)')
    axs[0, 0].set_title("Directional Accuracy (95% CI)")
    axs[0, 0].legend()
    set_y_range(axs[0, 0], metrics_df["Directional Accuracy"], include_baseline=[0.52, 0.55])

    # --- Plot 2: Pearson Correlation ---
    colors_r = ['red' if p > 0.05 else 'black' for p in metrics_df["Pearson pval"]]
    axs[0, 1].scatter(months, metrics_df["Pearson r"], color=colors_r)
    axs[0, 1].plot(months, metrics_df["Pearson r"], color='gray')
    axs[0, 1].axhline(0.05, color='red', linestyle='--', label='Min threshold (0.05)')
    axs[0, 1].set_title("Pearson Correlation")
    axs[0, 1].legend()
    set_y_range(axs[0, 1], metrics_df["Pearson r"], include_baseline=0.05)

    # --- Plot 3: WRMSE Improvement ---
    axs[1, 0].plot(months, metrics_df["WRMSE (%)"], marker='o')
    axs[1, 0].axhline(10, color='red', linestyle='--', label='Baseline 10%')
    axs[1, 0].set_title("WRMSE Improvement (%)")
    axs[1, 0].legend()
    set_y_range(axs[1, 0], metrics_df["WRMSE (%)"], include_baseline=10)

    # --- Plot 4: WMZTAE Improvement ---
    axs[1, 1].plot(months, metrics_df["WMZTAE (%)"], marker='o')
    axs[1, 1].axhline(20, color='red', linestyle='--', label='Baseline 20%')
    axs[1, 1].set_title("WMZTAE Improvement (%)")
    axs[1, 1].legend()
    set_y_range(axs[1, 1], metrics_df["WMZTAE (%)"], include_baseline=20)

    # -- Final Styling --
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel("Month")
        ax.set_ylabel("Score")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()