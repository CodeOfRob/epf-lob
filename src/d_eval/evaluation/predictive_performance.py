import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter, PercentFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

from .style import save_plot, get_model_style


def calc_statistical_metrics_split(
        predictions_scaled: dict,
        y_true_scaled: np.ndarray,
        predictions_raw: dict,
        y_true_raw: np.ndarray,
        timestamps: pd.Series,
        hac_lags: int = 30,
        baseline_model_name: str = "Baseline"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Berechnet statistische Metriken, führt DM-Tests durch und gibt ZWEI DataFrames zurück.

    Returns:
        df_error:   Enthält MAE, RMSE, W_MAE, rMAE (Fokus: Wie groß ist der Fehler?)
        df_quality: Enthält R2, HitRatios, Bias (Fokus: Wie gut ist die Struktur/Richtung?)
    """
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    model_names = list(predictions_scaled.keys())

    # --- 1. SAFE NUMPY CONVERSION ---
    def to_safe_array(data):
        if hasattr(data, "to_numpy"):
            return data.to_numpy(dtype=np.float64).ravel()
        return np.asarray(data, dtype=np.float64).ravel()

    y_true_s = to_safe_array(y_true_scaled)
    y_true_r = to_safe_array(y_true_raw)

    # Konstanten
    abs_true_r = np.abs(y_true_r)
    sum_abs_true = np.sum(abs_true_r)
    mask_nonzero = y_true_r != 0

    # Baseline MAE für rMAE
    if baseline_model_name in predictions_raw:
        base_pred = to_safe_array(predictions_raw[baseline_model_name])
        mae_baseline = mean_absolute_error(y_true_r, base_pred)
    else:
        mae_baseline = mean_absolute_error(y_true_r, np.zeros_like(y_true_r))

    # --- 2. METRIKEN BERECHNEN ---
    results = []

    for name in tqdm(model_names, desc="Calculating Metrics"):
        y_pred_s = to_safe_array(predictions_scaled[name])
        y_pred_r = to_safe_array(predictions_raw[name])

        # A) Scaled
        mae_s = mean_absolute_error(y_true_s, y_pred_s)
        rmse_s = np.sqrt(mean_squared_error(y_true_s, y_pred_s))
        r2_s = r2_score(y_true_s, y_pred_s)

        # B) Raw / Euro
        mae_r = mean_absolute_error(y_true_r, y_pred_r)
        rmse_r = np.sqrt(mean_squared_error(y_true_r, y_pred_r))
        r2_r = r2_score(y_true_r, y_pred_r)

        # rMAE
        rmae = mae_r / mae_baseline if mae_baseline != 0 else np.nan

        # Bias
        bias_r = np.mean(y_true_r - y_pred_r)

        # Hit Ratios
        sign_true = np.sign(y_true_r[mask_nonzero])
        sign_pred = np.sign(y_pred_r[mask_nonzero])
        dir_match = (sign_true == sign_pred).astype(float)

        hit_ratio = np.mean(dir_match)

        results.append({
            "Model": name,
            "MAE[S]": mae_s, "MAE[€]": mae_r,
            "RMSE[S]": rmse_s, "RMSE[€]": rmse_r,
            "rMAE": rmae,
            # Trennung hier gedanklich schon vollzogen:
            "R2[S]": r2_s, "R2[€]": r2_r,
            "Bias[€]": bias_r,
            "HitRatio": hit_ratio,
        })

    df = pd.DataFrame(results).set_index("Model")

    # --- 3. LEADER DEFINIEREN ---
    metric_logic = {
        "MAE[S]": False, "RMSE[S]": False, "R2[S]": True,
        "MAE[€]": False, "RMSE[€]": False, "R2[€]": True,
        "rMAE": False,
        "Bias[€]": "abs_min",
        "HitRatio": True,
    }

    leaders = {}
    for col, criteria in metric_logic.items():
        if criteria == "abs_min":
            leaders[col] = df[col].abs().idxmin()
        elif criteria:
            leaders[col] = df[col].idxmax()
        else:
            leaders[col] = df[col].idxmin()

    # --- 4. SIGNIFIKANZ-TESTS (DM-Test) ---
    final_df = df.copy().astype(str)  # Wir arbeiten auf einem String-Copy weiter

    # Leader Vektoren vorbereiten
    leader_vectors = {}
    for col, leader_name in leaders.items():
        use_scaled = "[S]" in col
        raw_preds = predictions_scaled[leader_name] if use_scaled else predictions_raw[leader_name]
        leader_pred = to_safe_array(raw_preds)
        true_vals = y_true_s if use_scaled else y_true_r

        leader_vectors[col] = {"pred": leader_pred, "true": true_vals}

    for col in tqdm(df.columns, desc="Significance Testing"):
        leader_name = leaders[col]

        # Bias: Keine Signifikanz, nur Wert formatieren
        if "Bias" in col:
            val = df.loc[leader_name, col]
            final_df.loc[leader_name, col] = f"{val:.4f}†"
            for m in model_names:
                if m != leader_name:
                    final_df.loc[m, col] = f"{df.loc[m, col]:.4f}"
            continue

        l_pred = leader_vectors[col]["pred"]
        y_vec = leader_vectors[col]["true"]

        # Leader Loss berechnen
        if "MAE" in col or "rMAE" in col:
            loss_leader = np.abs(y_vec - l_pred)
        elif "RMSE" in col or "R2" in col:
            loss_leader = (y_vec - l_pred) ** 2
        elif "HitRatio" in col:
            mask = y_vec != 0
            y_masked = y_vec[mask]
            l_masked = l_pred[mask]
            miss_leader = (np.sign(y_masked) != np.sign(l_masked)).astype(float)
            loss_leader = miss_leader

        # Loop über Modelle
        for model_name in model_names:
            val = df.loc[model_name, col]

            if model_name == leader_name:
                suffix = "†"
                fmt = "{:.2%}" if "Ratio" in col else "{:.4f}"
                final_df.loc[model_name, col] = (fmt + suffix).format(val)
                continue

            # Model Preds
            use_scaled = "[S]" in col
            raw_m_preds = predictions_scaled[model_name] if use_scaled else predictions_raw[model_name]
            m_pred = to_safe_array(raw_m_preds)

            # Model Loss
            if "MAE" in col or "rMAE" in col:
                loss_model = np.abs(y_vec - m_pred)
            elif "RMSE" in col or "R2" in col:
                loss_model = (y_vec - m_pred) ** 2
            elif "HitRatio" in col:
                y_masked = y_vec[mask]
                m_masked = m_pred[mask]
                miss_model = (np.sign(y_masked) != np.sign(m_masked)).astype(float)
                loss_model = miss_model

            # Diebold-Mariano Differenz
            d = loss_model - loss_leader

            # HAC Test
            if "HitRatio" in col:
                ols = sm.OLS(d, np.ones_like(d)).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
            else:
                # Group by Timestamp für korrekte Zeitreihen-Struktur
                ts_values = timestamps.values if hasattr(timestamps, "values") else timestamps
                temp_df = pd.DataFrame({'d': d, 'ts': ts_values})
                d_agg = temp_df.groupby('ts')['d'].mean()
                ols = sm.OLS(d_agg, np.ones(len(d_agg))).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})

            t_stat = ols.tvalues[0]
            p_val = ols.pvalues[0]

            # Einseitiger Test (H0: Modell ist nicht schlechter als Leader)
            if t_stat > 0:
                p_one_sided = p_val / 2
            else:
                p_one_sided = 1.0

            stars = ""
            if p_one_sided < 0.01:
                stars = "***"
            elif p_one_sided < 0.05:
                stars = "**"
            elif p_one_sided < 0.10:
                stars = "*"

            fmt = "{:.2%}" if "Ratio" in col else "{:.4f}"
            final_df.loc[model_name, col] = (fmt + "{}").format(val, stars)

    # --- 5. SPLIT IN ZWEI DATAFRAMES (Option 2) ---

    # Tabelle A: Error Metrics (Magnitude)
    # Alles was Fehler misst (MAE, RMSE, rMAE, W_MAE) - egal ob scaled oder raw
    cols_error = [
        "MAE[S]", "MAE[€]",
        "RMSE[S]", "RMSE[€]",
        "rMAE"
    ]

    # Tabelle B: Quality Metrics (Direction & Fit)
    # Alles was Richtung, Erklärungskraft oder Bias misst
    cols_quality = [
        "R2[S]", "R2[€]",
        "HitRatio",
        "Bias[€]"
    ]

    # Sicherstellen, dass die Spalten in der gewünschten Reihenfolge ausgegeben werden
    df_error = final_df[cols_error].copy().sort_values(by="rMAE")
    df_quality = final_df[cols_quality].copy().sort_values(by="HitRatio", ascending=False)

    return df_error, df_quality


def plot_reliability_density(
        predictions_raw: dict,
        y_true_raw: np.ndarray,

        # --- BINNING PARAMETER ---
        bin_step_rel=0.05,  # Für Reliability
        bin_step_acc=0.10,  # Für Directional Accuracy

        # Panel D (Coverage)
        n_thresholds_cov=50,

        # Limits (Optional)
        limits_A_acc=None,  # Jetzt Accuracy (links)
        limits_B_cal=None,  # Jetzt Reliability (rechts)
        limits_C_dens=None,
        limits_D_cov=None,

        save_name=None
):
    model_names = list(predictions_raw.keys())
    y_true = np.array(y_true_raw).flatten()

    # Berechne Anzahl der Bins
    n_bins_rel = int(1 / bin_step_rel)
    n_bins_acc = int(1 / bin_step_acc)

    print(f"Generiere Plots:")
    print(f" - Panel A (Accuracy):    {bin_step_acc * 100:.1f}% Schritte ({n_bins_acc} Bins)")
    print(f" - Panel B (Reliability): {bin_step_rel * 100:.1f}% Schritte ({n_bins_rel} Bins)")

    # --- STYLE SETUP ---
    try:
        palette, markers = get_model_style(model_names)
    except NameError:
        colors = sns.color_palette("husl", len(model_names))
        palette = dict(zip(model_names, colors))
        markers = {name: 'o' for name in model_names}

    # --- LAYOUT SETUP ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- TAUSCH DER POSITIONEN HIER ---
    ax_acc = axes[0, 0]  # Oben Links: Accuracy (vorher Rechts)
    ax_rel = axes[0, 1]  # Oben Rechts: Reliability (vorher Links)

    ax_den = axes[1, 0]  # Unten Links: Density
    ax_cov = axes[1, 1]  # Unten Rechts: Coverage

    # --- GLOBALE REFERENZEN ---
    all_preds = np.concatenate([p.flatten() for p in predictions_raw.values()])
    global_min, global_max = np.min(all_preds), np.max(all_preds)

    # ---------------------------------------------------------
    # STATIC ELEMENTS & LOKALE LEGENDEN
    # ---------------------------------------------------------

    # A (jetzt Accuracy): Random Guess Line (50%)
    line_rand = ax_acc.axhline(0.5, color='gray', linestyle=':', alpha=0.8,
                               linewidth=1.2, label='Random Guess')
    ax_acc.legend(handles=[line_rand], loc='lower right', frameon=False, fontsize=10)

    # B (jetzt Reliability): Perfect Calibration Line
    line_perf, = ax_rel.plot([global_min, global_max], [global_min, global_max],
                             color="black", linestyle="--", linewidth=1.0, alpha=0.5,
                             label="Perfect Calibration", zorder=0)
    ax_rel.legend(handles=[line_perf], loc='upper left', frameon=False, fontsize=10)

    # ---------------------------------------------------------
    # LOOP ÜBER MODELLE
    # ---------------------------------------------------------
    for name in model_names:
        y_pred = np.array(predictions_raw[name]).flatten()
        color = palette.get(name, 'black')
        marker = markers.get(name, 'o')

        df = pd.DataFrame({'pred': y_pred, 'true': y_true})

        # --- PANEL A: DIRECTIONAL ACCURACY (jetzt Links) ---
        try:
            df['rank_pct'] = df['pred'].rank(pct=True)
            df['hit'] = np.sign(df['pred']) == np.sign(df['true'])
            df['bin_id_acc'] = pd.qcut(df['pred'], q=n_bins_acc, labels=False, duplicates='drop')

            bin_stats_acc = df.groupby('bin_id_acc', observed=True).agg({'rank_pct': 'mean', 'hit': 'mean'})

            ax_acc.plot(bin_stats_acc['rank_pct'] * 100, bin_stats_acc['hit'],
                        marker=marker, markersize=5, color=color, label=name,
                        linewidth=1.5, alpha=0.8)
        except ValueError:
            pass

        # --- PANEL B: RELIABILITY (jetzt Rechts) ---
        try:
            df['bin_id_rel'] = pd.qcut(df['pred'], q=n_bins_rel, labels=False, duplicates='drop')
            bin_stats_rel = df.groupby('bin_id_rel', observed=True).agg({'pred': 'mean', 'true': 'mean'})

            ax_rel.plot(bin_stats_rel['pred'], bin_stats_rel['true'],
                        marker=marker, label=name, color=color,
                        linewidth=1.5, markersize=5, alpha=0.9)
        except ValueError:
            pass

        # --- PANEL C: DENSITY ---
        sns.kdeplot(y_pred, ax=ax_den, color=color, alpha=0.8,
                    linewidth=1.5, warn_singular=False, label=name)

        try:
            line = ax_den.get_lines()[-1]
            x_plot, y_plot = line.get_data()
            idx = np.argmax(y_plot)
            ax_den.plot(x_plot[idx], y_plot[idx],
                        marker=marker, color=color,
                        markersize=5, linestyle='None')
        except IndexError:
            pass

        # --- PANEL D: SIGNAL COVERAGE ---
        max_thresh = np.percentile(np.abs(y_pred), 99.5)
        thresholds = np.linspace(1e-4, max_thresh, n_thresholds_cov)
        coverage = [np.mean(np.abs(y_pred) > t) * 100 for t in thresholds]

        ax_cov.plot(thresholds, coverage, marker=marker, color=color, label=name,
                    linewidth=1.5, alpha=0.8, markevery=5, markersize=5)

    # ---------------------------------------------------------
    # FORMATIERUNG
    # ---------------------------------------------------------

    # PANEL A: Directional Accuracy (Links)
    ax_acc.set_title(f"A: Directional Accuracy ({bin_step_acc * 100:.0f}% Q-Bins)", loc='left', fontweight='bold')
    ax_acc.set_xlabel("Prediction Strength Percentile")
    ax_acc.set_ylabel("Directional Accuracy (Hit-Ratio)")
    ax_acc.grid(True, linestyle=':', alpha=0.4)
    ax_acc.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax_acc.set_xlim(0, 100)
    if limits_A_acc: ax_acc.set_ylim(limits_A_acc)

    # PANEL B: Reliability (Rechts)
    ax_rel.set_title(f"B: Reliability Diagram ({bin_step_rel * 100:.0f}% Q-Bins)", loc='left', fontweight='bold')
    ax_rel.set_xlabel(r"Mean Predicted Price $\mathbb{E}[\hat{y}]$")
    ax_rel.set_ylabel(r"Mean Observed Price $\mathbb{E}[y]$")
    ax_rel.grid(True, linestyle=':', alpha=0.4)
    if limits_B_cal:
        ax_rel.set_xlim(limits_B_cal[0])
        ax_rel.set_ylim(limits_B_cal[1])

    # PANEL C: Density
    ax_den.set_title("C: Forecast Distribution (Density)", loc='left', fontweight='bold')
    ax_den.set_xlabel(r"Predicted Price $\hat{y}$")
    ax_den.set_ylabel("Density")
    ax_den.grid(True, linestyle=':', alpha=0.4)
    if limits_C_dens:
        ax_den.set_xlim(limits_C_dens[0])
        ax_den.set_ylim(limits_C_dens[1])

    # PANEL D: Coverage
    ax_cov.set_title("D: Signal Coverage (Log-Scale)", loc='left', fontweight='bold')
    ax_cov.set_xlabel(r"Absolute Threshold $|\hat{y}|$")
    ax_cov.set_ylabel("Samples exceeding Threshold [%]")
    ax_cov.set_yscale('log')
    ax_cov.yaxis.set_major_formatter(ScalarFormatter())
    ax_cov.grid(True, linestyle=':', alpha=0.4, which='both')
    if limits_D_cov:
        if limits_D_cov[0] is not None: ax_cov.set_xlim(limits_D_cov[0])
        if limits_D_cov[1] is not None: ax_cov.set_ylim(limits_D_cov[1])

    # ---------------------------------------------------------
    # GLOBALE LEGENDE
    # ---------------------------------------------------------
    handles, labels = ax_cov.get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc='lower center', bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(model_names), 6), fontsize=11, frameon=False
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_name:
        save_plot_folder = "figures/pred"
        save_plot(fig, save_name, save_plot_folder)
    plt.show()


def test_regime_analysis(
        predictions_raw: dict,
        y_true_raw: np.ndarray,
        X_test_raw: pd.DataFrame,
        snapshot_times: pd.Series,
        baseline_name: str = "LASSO",
        resample_rule='D',
        rolling_window=5,
        events=None,
        save_name=None
):
    # --- 0. SETUP & STYLING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    if events is None:
        events = {
            "IDCC": ("2024-05-22", "#d62728"),
            "IDA": ("2024-06-13", "#ff7f0e")
        }

    model_names = list(predictions_raw.keys())
    # Hier wird angenommen, dass get_model_style existiert, sonst fallback auf husl
    try:
        palette, markers = get_model_style(model_names)
    except:
        colors = sns.color_palette("husl", len(model_names))
        palette = dict(zip(model_names, colors))
        markers = {name: 'o' for name in model_names}

    times = pd.to_datetime(pd.Series(snapshot_times)).dt.tz_localize(None)

    # --- 1. DATENBERECHNUNG ---
    df_calc = pd.DataFrame(index=times)
    y_true = np.array(y_true_raw).flatten()
    for name in model_names:
        y_pred = np.array(predictions_raw[name]).flatten()
        df_calc[f'{name}_err'] = np.abs(y_pred - y_true)

    df_features = X_test_raw.copy()
    df_features.index = times
    df_features['avg_slope'] = (df_features['orderbook_slope_ask'] + df_features['orderbook_slope_bid']) / 2

    # --- 2. AGGREGATION ---
    df_mae_res = df_calc.resample(resample_rule).mean().rolling(window=rolling_window, min_periods=1).mean()
    df_feat_res = df_features.resample(resample_rule).mean().rolling(window=rolling_window, min_periods=1).mean()

    # rMAE Berechnung
    df_rmae = pd.DataFrame(index=df_mae_res.index)
    base_col = f'{baseline_name}_err'
    for name in model_names:
        df_rmae[name] = df_mae_res[f'{name}_err'] / df_mae_res[base_col].replace(0, np.nan)

    # Regime Maske (Slope > Median)
    slope_threshold = df_feat_res['avg_slope'].median()
    high_slope_mask = df_feat_res['avg_slope'] > slope_threshold

    # --- 3. PLOTTING ---
    total_rows = 4
    height_ratios = [2, 0.75, 0.75, 0.75]

    fig, axes = plt.subplots(
        total_rows, 1,
        figsize=(12, 2 + 2.5 * 3),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.15}
    )

    # --- PANEL 1: Relative Performance ---
    ax = axes[0]
    for name in model_names:
        if name == baseline_name: continue
        series = df_rmae[name].dropna()
        ax.plot(series.index, series.values, label=name.upper(), color=palette.get(name),
                linewidth=2.0 if 'cb' in name.lower() else 1.5,
                marker=markers.get(name), markevery=max(1, len(series) // 20), alpha=0.9, ms=5)

    ax.axhline(1.0, color=palette.get(name), linestyle='--', linewidth=1.5, label=f'Baseline ({baseline_name.upper()})')
    # Publikationsreife Beschriftung
    ax.set_ylabel(f"Ratio vs {baseline_name}", fontsize=10)
    ax.set_title(f"A: Relative Forecast Error rMAE (Rolling {rolling_window}{resample_rule} Mean)", fontsize=12,
                 fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', framealpha=0.9, ncol=3)

    # Fokus auf relevanten Bereich (z.B. +/- 1% vom Baseline-Wert)
    ax.set_ylim(df_rmae.min().min() * 0.999, df_rmae.max().max() * 1.001)

    # --- PANEL 2: Absolute MAE (NEU) ---
    ax = axes[1]
    # Plot der Range (Min/Max MAE aller Modelle)
    mae_min = df_mae_res.min(axis=1)
    mae_max = df_mae_res.max(axis=1)
    mae_avg = df_mae_res.mean(axis=1)

    ax.fill_between(df_mae_res.index, mae_min, mae_max, color='gray', alpha=0.2, label='Model Range')
    ax.plot(df_mae_res.index, mae_avg, color='black', linewidth=1.5, label="Avg. MAE")

    ax.set_ylabel("EUR/MWh", fontsize=10)
    ax.set_title("B: Absolute Forecast Error (MAE)", fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.9, ncol=3)

    # --- PANEL 3: Volatility ---
    ax = axes[2]
    ax.plot(df_feat_res.index, df_feat_res['mid_price_return_prev_5min_RV_1800s'], color='#555555', linewidth=1.5,
            label="mid_price_return_prev_5min_RV_1800s")
    ax.set_ylabel("EUR/MWh", fontsize=10)
    ax.set_title("C: Market Volatility", fontsize=11, fontweight='bold',
                 loc='left')
    ax.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.9, ncol=3)

    # --- PANEL 3: Order Book Slope ---
    ax = axes[3]
    ax.plot(df_feat_res.index, df_feat_res['avg_slope'], color='#1f78b4', linewidth=1.5, label="avg_orderbook_slope")
    ax.set_ylabel("Slope Factor", fontsize=10)
    ax.set_title("D: Liquidity Sensitivity", fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.9, ncol=3)

    # --- GLOBAL SHADING, EVENTS & FORMATTING ---
    for i, ax in enumerate(axes):
        # Events mit sauberem Text-Placement
        ylim = ax.get_ylim()
        for label, (date_str, color) in events.items():
            try:
                date_obj = pd.Timestamp(date_str)
                if date_obj >= times.min() and date_obj <= times.max():
                    ax.axvline(date_obj, color=color, linestyle=':', linewidth=2, alpha=0.8, zorder=10)
                    if i == 0:
                        ax.text(date_obj, ylim[1], f' {label}', color=color, va='top', rotation=90,
                                fontsize=10, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            except:
                pass

        ax.grid(True, linestyle='--', alpha=0.4)

    # X-Achse Formatierung
    axes[-1].set_xlabel("Time Period (2024)", fontweight='bold', fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.tight_layout()

    if save_name:
        save_plot_folder = "figures/pred"
        save_plot(fig, save_name, save_plot_folder)
    plt.show()


def lifecycle_regime_analysis(
        predictions_raw: dict,
        y_true_raw: np.ndarray,
        X_test_raw: pd.DataFrame,
        feature_config: dict,
        ttd_series: pd.Series,
        baseline_name: str = "lasso",
        bin_step: float = 10.0,
        ttd_events: dict = None,
        save_name=None
):
    # --- 1. DATA PREPARATION ---
    model_names = list(predictions_raw.keys())
    df_master = pd.DataFrame({
        'ttd': np.array(ttd_series),
        'y_true': np.array(y_true_raw).flatten()
    })

    for feat_col in X_test_raw.select_dtypes(include=[np.number]).columns:
        df_master[f"feat_{feat_col}"] = X_test_raw[feat_col].values

    for name in model_names:
        pred = np.array(predictions_raw[name]).flatten()
        df_master[f"err_{name}"] = np.abs(pred - df_master['y_true'])

    df_master['ttd_bin'] = (df_master['ttd'] / bin_step).round() * bin_step
    grouped = df_master.groupby('ttd_bin')

    feat_cols = [c for c in df_master.columns if c.startswith("feat_")]
    df_feat_agg = grouped[feat_cols].mean()
    df_feat_agg.columns = [c.replace("feat_", "") for c in df_feat_agg.columns]

    df_mae = pd.DataFrame({name: grouped[f"err_{name}"].mean() for name in model_names})
    df_mae.sort_index(inplace=True)
    df_rmae = df_mae.div(df_mae[baseline_name], axis=0)

    # --- 2. PLOTTING SETUP MIT HEIGHT RATIOS ---
    num_feature_panels = len(feature_config)
    total_panels = 2 + num_feature_panels

    # Definition der Höhenverhältnisse:
    # Performance-Panels (A & B) sind doppelt so hoch wie die Feature-Panels (C+)
    h_ratios = [2, 0.5] + [0.5] * num_feature_panels

    fig, axes = plt.subplots(
        total_panels, 1,
        figsize=(12, 2 + 2.5 * num_feature_panels),
        sharex=True,
        gridspec_kw={'height_ratios': h_ratios, 'hspace': 0.3}
    )

    alphabet = string.ascii_uppercase
    palette, markers = get_model_style(model_names)
    feat_colors = sns.color_palette("tab10")

    # --- PANEL A: Relative MAE ---
    ax0 = axes[0]
    for name in model_names:
        if name == baseline_name: continue
        ax0.plot(df_rmae.index, df_rmae[name], label=name, color=palette[name], lw=1.5, marker=markers[name],
                 ms=5)
    ax0.axhline(1.0, color=palette.get(baseline_name, "black"), linestyle='--', label=f'Baseline ({baseline_name})')
    ax0.set_title(f"{alphabet[0]}: Relative Forecast Error rMAE (Rolling {bin_step}min Mean)", loc='left', fontsize=12,
                  fontweight='bold')
    ax0.set_ylabel(f"Ratio vs. {baseline_name}")
    ax0.legend(loc='lower left', fontsize=9, frameon=True, facecolor='white', framealpha=0.9, ncol=3)

    # --- PANEL B: Absolute MAE ---
    ax1 = axes[1]
    mean_mae = df_mae.mean(axis=1)
    ax1.fill_between(df_mae.index, df_mae.min(axis=1), df_mae.max(axis=1), color='gray', alpha=0.15)
    ax1.plot(mean_mae.index, mean_mae, color='black', lw=2, label='Avg. Error')
    ax1.set_title(f"{alphabet[1]}: Absolute Forecast Error (MAE)", loc='left', fontsize=11, fontweight='bold')
    ax1.set_ylabel("EUR/MWh")
    ax1.legend(loc='upper left', fontsize=8, frameon=True)

    # --- PANELS C+: Features ---
    for i, (title, cfg) in enumerate(feature_config.items()):
        ax = axes[i + 2]
        valid_features = [f for f in cfg['cols'] if f in df_feat_agg.columns]
        for j, feat_name in enumerate(valid_features):
            ax.plot(df_feat_agg.index, df_feat_agg[feat_name], label=feat_name, color=feat_colors[j % 10], lw=1.2)

        ax.set_title(f"{alphabet[i + 2]}: {title}", loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel(cfg.get('unit', '-'), fontsize=9)
        ax.legend(loc='upper left', fontsize=8, frameon=True)

    # --- GLOBAL ADJUSTMENTS ---
    if ttd_events is None:
        ttd_events = {
            "SIDC GC (T-60)": (60, "gray"),
            "CZ GC (T-30)": (30, "orange")
        }

    for ax in axes:
        ax.invert_xaxis()
        ax.grid(True, linestyle=':', alpha=0.5)
        for label, (pos, color) in ttd_events.items():
            ax.axvline(pos, color=color, linestyle=':', alpha=0.8, lw=1.2)
            if ax == axes[0]:
                ax.text(pos - 1, ax.get_ylim()[1], f" {label}", color=color, va='top', rotation=90, fontsize=8,
                        fontweight='bold')

    axes[-1].set_xlabel("Time to Delivery [Minutes]", fontweight='bold')

    if save_name:
        save_plot_folder = "figures/pred"
        save_plot(fig, save_name, save_plot_folder)
    plt.show()


from typing import Dict, Literal
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dynamic_regression_rmae_robustness(
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        df_metadata: pd.DataFrame,
        features_to_analyze: list,
        baseline_name: str = "LASSO",
        n_bins_quantile: int = 15,
        n_cols: int = 2,
        save_name: str = None
):
    # 0. Style Setup
    model_names = list(predictions.keys())
    # Nutze Standard-Farben falls get_model_style nicht verfügbar
    try:
        from src.eval.evaluation.style import get_model_style
        palette, markers = get_model_style(model_names)
    except ImportError:
        colors = sns.color_palette("husl", len(model_names))
        palette = dict(zip(model_names, colors))
        markers = {name: 'o' for name in model_names}

    # 1. Daten-Vorbereitung
    df = df_metadata.copy()
    df['y_true'] = np.array(y_true, dtype=np.float64)

    num_plots = len(features_to_analyze)
    n_rows = int(np.ceil(num_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows + 1), squeeze=False)
    axes = axes.flatten()

    handles, labels = None, None

    for idx, feature_col in tqdm(enumerate(features_to_analyze), total=num_plots, desc="rMAE Robustness"):
        ax = axes[idx]
        if feature_col not in df.columns: continue

        # --- INTELLIGENTER TYPE CHECK ---
        raw_col = df[feature_col]
        is_datetime = pd.api.types.is_datetime64_any_dtype(raw_col)

        if pd.api.types.is_numeric_dtype(raw_col) and not is_datetime:
            current_feat_data = raw_col.astype(np.float32)
        else:
            current_feat_data = raw_col

        is_bool_binary = current_feat_data.dtype in ['bool', 'int8', 'int16'] or current_feat_data.nunique() <= 2

        bin_col_name = f"{feature_col}_bin"
        plot_type = 'line'
        xtick_labels = None

        try:
            # BINNING
            if is_datetime or is_bool_binary or (not is_datetime and current_feat_data.nunique() <= 30):
                df[bin_col_name] = current_feat_data
                if is_bool_binary or (not is_datetime and current_feat_data.nunique() <= 4):
                    plot_type = 'bar'
                    if is_bool_binary:
                        xtick_labels = ['Off/False', 'Peak/True']
                        df[bin_col_name] = current_feat_data.astype(str).replace(
                            {'0': 'Off/False', '1': 'Peak/True', '0.0': 'Off/False', '1.0': 'Peak/True'})
            else:
                q_res = pd.qcut(current_feat_data.astype(np.float32), q=n_bins_quantile, labels=False, retbins=True,
                                duplicates='drop')
                df[bin_col_name] = q_res[0].astype(int)
                bounds = q_res[1]
                xtick_labels = [f"{bounds[i]:.1f}—{bounds[i + 1]:.1f}" for i in range(len(bounds) - 1)]
                plot_type = 'line'

            # --- rMAE BERECHNUNG ---
            # Liste für Aggregation
            results = []
            for name in model_names:
                df_temp = df.copy()
                df_temp['abs_err'] = np.abs(np.array(predictions[name]).flatten() - df_temp['y_true'])
                # MAE pro Bin
                bin_mae = df_temp.groupby(bin_col_name)['abs_err'].mean()
                results.append(pd.DataFrame({'Model': name, 'MAE': bin_mae}).reset_index())

            df_plot_all = pd.concat(results)

            # Pivotieren für rMAE (Ratio vs Baseline)
            df_pivot = df_plot_all.pivot(index=bin_col_name, columns='Model', values='MAE')
            if baseline_name in df_pivot.columns:
                df_rmae = df_pivot.div(df_pivot[baseline_name], axis=0).reset_index()
                df_plot_final = df_rmae.melt(id_vars=bin_col_name, var_name='Model', value_name='rMAE')
            else:
                # Fallback falls Baseline fehlt
                df_plot_final = df_plot_all.rename(columns={'MAE': 'rMAE'})

            # PLOTTING
            if plot_type == 'line':
                sns.lineplot(data=df_plot_final, x=bin_col_name, y='rMAE', hue='Model', style='Model',
                             palette=palette, markers=True, markersize=7, ax=ax, dashes=False, linewidth=2)
            else:
                sns.barplot(data=df_plot_final, x=bin_col_name, y='rMAE', hue='Model', palette=palette, ax=ax)

            if xtick_labels:
                ax.set_xticks(range(len(xtick_labels)))
                ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)

            ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_title(f"{feature_col}", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"rMAE (vs {baseline_name})")
            ax.set_xlabel("")
            ax.grid(True, alpha=0.3)

            if handles is None: handles, labels = ax.get_legend_handles_labels()
            if ax.get_legend(): ax.get_legend().remove()

        except Exception as e:
            ax.set_title(f"ERROR in {feature_col}", color='red')
            print(f"Error in {feature_col}: {e}")

    # Aufräumen & Legend
    for i in range(num_plots, len(axes)): fig.delaxes(axes[i])
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(predictions), 5),
                   bbox_to_anchor=(0.5, 0.03), frameon=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.07, 1, 0.98])
    if save_name:
        save_plot_folder = "figures/pred"
        save_plot(fig, save_name, save_plot_folder)
    plt.show()
    return fig


def plot_payoff_analysis(
        predictions_raw: dict,
        y_true_raw: np.ndarray,
        baseline_name='sgd',
        n_quantiles=20,
        save_name=None
):
    """
    Erstellt ein publikationsreifes 2-Panel-Plot für die Masterarbeit.
    """

    # --- 0. STYLE SETUP FÜR PUBLIKATION ---
    # Setzt globale Parameter für bessere Lesbarkeit in Dokumenten
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'font.family': 'sans-serif'  # Oder 'serif' für LaTeX-Look
    })

    model_names = list(predictions_raw.keys())
    y_true = np.array(y_true_raw)
    abs_true = np.abs(y_true)

    # --- 1. DATEN VORBEREITUNG (Quantile) ---
    try:
        q_bins = pd.qcut(abs_true, q=n_quantiles, duplicates='drop')
    except ValueError:
        q_bins = pd.qcut(abs_true, q=10, duplicates='drop')

    df_bins = pd.DataFrame({'abs_val': abs_true, 'bin': q_bins})
    bin_stats = df_bins.groupby('bin', observed=True)['abs_val'].mean()
    bin_centers = bin_stats.values

    abs_ratios = {name: [] for name in model_names}
    valid_centers = []

    for bin_cat, bin_center in zip(bin_stats.index, bin_centers):
        mask_bin = (q_bins == bin_cat)
        if np.sum(mask_bin) < 5: continue

        current_bin_ratios = {}
        baseline_valid = False

        for name in model_names:
            y_pred = np.array(predictions_raw[name])
            abs_err = np.abs(y_pred - y_true)
            correct = np.sign(y_pred) == np.sign(y_true)

            mae_c = np.mean(abs_err[mask_bin & correct]) if np.sum(mask_bin & correct) > 0 else np.nan
            mae_w = np.mean(abs_err[mask_bin & (~correct)]) if np.sum(mask_bin & (~correct)) > 0 else np.nan

            # Ratio: Reward / Risk
            if not np.isnan(mae_c) and not np.isnan(mae_w) and mae_w > 0:
                ratio = mae_c / mae_w
            else:
                ratio = np.nan

            current_bin_ratios[name] = ratio
            if name == baseline_name and not np.isnan(ratio) and ratio > 0:
                baseline_valid = True

        if baseline_valid:
            valid_centers.append(bin_center)
            for name in model_names:
                abs_ratios[name].append(current_bin_ratios[name])

    # --- 2. PLOTTING SETUP ---
    # Etwas höher für mehr Platz
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0.15})

    # Fallback für Farben/Marker
    try:
        palette, markers = get_model_style(model_names)
    except:
        colors = sns.color_palette("husl", len(model_names))
        palette = dict(zip(model_names, colors))
        markers = {name: 'o' for name in model_names}

    # ==========================================
    # PANEL A: RELATIVE SKILL (vs Baseline)
    # ==========================================
    # Baseline Referenz
    ax_top.axhline(1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Baseline ({baseline_name})')
    base_vals = np.array(abs_ratios[baseline_name])

    for name in model_names:
        if name == baseline_name: continue

        model_vals = np.array(abs_ratios[name])
        relative_vals = model_vals / base_vals
        mask = np.isfinite(relative_vals)

        ax_top.plot(np.array(valid_centers)[mask], relative_vals[mask],
                    marker=markers.get(name, 'o'), color=palette.get(name),
                    linewidth=2.0, label=name, markersize=5)

        # Highlight Fill (Dezent)
        ax_top.fill_between(np.array(valid_centers)[mask], 1.0, relative_vals[mask],
                            where=(relative_vals[mask] > 1.0), color=palette.get(name), alpha=0.1)

    ax_top.set_ylabel(f"Relative Asymmetry\n(Model / {baseline_name})", fontweight='bold')
    ax_top.set_title("Panel A: Relative Skill vs. Baseline", loc='left', fontweight='bold', pad=10)

    # Annotationen Panel A
    # Statische Positionierung via transform=ax.transAxes (unabhängig von Daten)
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#cccccc', linewidth=0.8)

    textstr = '\n'.join((
        r'$\bf{Metric:}$ Relative Payoff Ratio',
        r'Curve $>$ 1.0: Model has better risk/reward',
        r'profile than baseline as volatility rises.'
    ))
    ax_top.text(0.02, 0.95, textstr, transform=ax_top.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Dynamische Labels rechts
    ax_top.text(valid_centers[-1], 1.03, "Outperforming\nBaseline", color='#2ca02c', ha='right', va='bottom',
                fontsize=9, fontweight='bold')
    ax_top.text(valid_centers[-1], 0.97, "Underperforming\nBaseline", color='#d62728', ha='right', va='top', fontsize=9,
                fontweight='bold')

    # ==========================================
    # PANEL B: ABSOLUTE PROFITABILITY
    # ==========================================
    # Zonen (Dezenter für Druck)
    ax_bot.axhspan(1.0, 10.0, color='#2ca02c', alpha=0.08)  # Grün
    ax_bot.axhspan(0.0, 1.0, color='#d62728', alpha=0.08)  # Rot
    ax_bot.axhline(1.0, color='#555555', linestyle='-', linewidth=1.5, alpha=0.8)

    for name in model_names:
        vals = np.array(abs_ratios[name])
        mask = np.isfinite(vals)
        ax_bot.plot(np.array(valid_centers)[mask], vals[mask],
                    marker=markers.get(name, 'o'), color=palette.get(name),
                    linewidth=2.0, label=name, markersize=5)

    ax_bot.set_ylabel("Absolute Payoff Ratio\n(MAE Correct / MAE Wrong)", fontweight='bold')
    ax_bot.set_title("Panel B: Absolute Profitability", loc='left', fontweight='bold', pad=10)
    ax_bot.set_xlabel("Market Volatility Regime (Mean Absolute Magnitude) [EUR/MWh]", fontweight='bold', labelpad=10)

    # Annotationen Panel B
    # Text innerhalb der Zonen
    ax_bot.text(0.02, 0.92, "PROFITABLE ZONE (Wins > Losses)", transform=ax_bot.transAxes,
                color='#1a701a', fontweight='bold', fontsize=9)
    ax_bot.text(0.02, 0.08, "UNPROFITABLE ZONE (Losses > Wins)", transform=ax_bot.transAxes,
                color='#a61e1e', fontweight='bold', fontsize=9)

    # Marktphasen Pfeile (unten fixiert)
    ax_bot.text(0.01, -0.16, "← Low Volatility (Noise)", transform=ax_bot.transAxes, ha='left', style='italic',
                color='#666666')
    ax_bot.text(0.99, -0.16, "Extreme Volatility (Chaos) →", transform=ax_bot.transAxes, ha='right', style='italic',
                color='#666666')

    # --- FORMATIERUNG & LAYOUT ---

    # Log Skala für X
    ax_bot.set_xscale('log')
    # Erzwinge normale Zahlenformatierung (1, 10, 100) statt 10^1
    for ax in [ax_top, ax_bot]:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis='x', which='minor', bottom=False)  # Keine kleinen Ticks beschriften

    # Limits dynamisch aber sinnvoll setzen
    # Oben: Fokus auf 1.0
    ax_top.set_ylim(0.85,
                    max(1.15, np.nanmax([np.nanmax(abs_ratios[n]) / np.nanmax(base_vals) for n in model_names]) * 1.05))
    # Unten: Fokus auf Threshold
    ax_bot.set_ylim(0.2, 1.2)  # Oder weiter, je nach Daten

    # Grid
    for ax in [ax_top, ax_bot]:
        ax.grid(True, which="major", linestyle='-', alpha=0.3, color='#bbbbbb')
        ax.grid(True, which="minor", linestyle=':', alpha=0.2, color='#cccccc')

    # Gemeinsame Legende UNTEN (Clean)
    # Wir sammeln Handles von Panel B (enthält alle Modelle) + Baseline von Panel A
    handles_b, labels_b = ax_bot.get_legend_handles_labels()
    handles_a, labels_a = ax_top.get_legend_handles_labels()

    # Dictionary für einzigartige Labels (Baseline nur einmal)
    unique_legend = {}
    # Erst Baseline hinzufügen (damit sie vorne steht)
    for h, l in zip(handles_a, labels_a):
        if 'Baseline' in l:
            unique_legend[l] = h
    # Dann Modelle
    for h, l in zip(handles_b, labels_b):
        unique_legend[l] = h

    fig.legend(unique_legend.values(), unique_legend.keys(),
               loc='lower center', ncol=len(unique_legend),
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, fancybox=True, shadow=False, borderpad=0.5)

    # Layout Anpassung für Legende unten
    plt.subplots_adjust(bottom=0.12, top=0.95)

    if save_name:
        save_plot_folder = "figures/pred"
        save_plot(fig, save_name, save_plot_folder)

    plt.show()
