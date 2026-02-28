import math
from typing import Callable, List
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks
from tqdm import tqdm

from .style import save_plot, get_model_style
from .tradingEvaluator import TradingEvaluator
from .tradingStrategyManager import TradingStrategyManager


def align_market_prices_to_execution_window(
        df_signals: pd.DataFrame,
        df_market_data: pd.DataFrame,
        latency_sec: int = 10,
        staleness_tolerance_sec: int = 30,
        best_bid_col: str = 'price_1_bid',
        best_ask_col: str = 'price_1_ask'
) -> pd.DataFrame:
    """
    Richtet Handelssignale an zeitverzögerten Ausführungspreisen aus.
    Implementiert die Latenz-Logik gemäß Sektion 4.3.3[cite: 1214].

    Returns:
        pd.DataFrame: Enthält Snapshot-Metadaten sowie vier Preispunkte:
            - signal_time_bid/ask: Liquidität zum Zeitpunkt t_0 (Inferenz).
            - execution_bid/ask: Liquidität zum Zeitpunkt t_0 + latency (Order-Eingang).
    """

    df_signals = df_signals.copy()
    df_market_snapshot = df_market_data[
        ['delivery_start', 'snapshot_times', best_bid_col, best_ask_col]
    ].sort_values('snapshot_times')

    df_signals['snapshot_times'] = pd.to_datetime(df_signals['snapshot_times'])
    df_market_snapshot['snapshot_times'] = pd.to_datetime(df_market_snapshot['snapshot_times'])

    # --- SCHRITT 1: Preise zum Entscheidungszeitpunkt (t_0) ---
    df_combined = pd.merge_asof(
        df_signals.sort_values('snapshot_times'),
        df_market_snapshot,
        on='snapshot_times',
        by='delivery_start',
        direction='backward'
    ).rename(columns={
        best_bid_col: 'signal_time_bid',
        best_ask_col: 'signal_time_ask'
    })

    # --- SCHRITT 2: Preise zum Ausführungszeitpunkt (t_0 + Latenz) ---
    df_combined['timestamp_at_execution'] = (
            df_combined['snapshot_times'] + pd.Timedelta(seconds=latency_sec)
    )

    df_combined = pd.merge_asof(
        df_combined.sort_values('timestamp_at_execution'),
        df_market_snapshot,
        left_on='timestamp_at_execution',
        right_on='snapshot_times',
        by='delivery_start',
        direction='backward',
        tolerance=pd.Timedelta(seconds=staleness_tolerance_sec),
        suffixes=('', '_market_sync')
    ).rename(columns={
        best_bid_col: 'exec_bid',
        best_ask_col: 'exec_ask'
    })

    return df_combined.drop(columns=['timestamp_at_execution', 'snapshot_times_market_sync'])


def build_backtest_foundation(
        df_signal_metadata: pd.DataFrame,
        df_raw_lob_data: pd.DataFrame,
        model_predictions: Dict[str, np.ndarray],
        execution_latency_sec: int = 10,
        price_staleness_limit_sec: int = 20
) -> pd.DataFrame:
    """
    Konstruiert das 'Analytical Substrate' für die ökonomische Evaluation[cite: 227].
    Integriert Modell-Inferenz, Entscheidungs-Liquidität und Ausführungs-Liquidität.

    Structure of Returned DataFrame:
        - Index: snapshot_times (datetime)
        - delivery_start: Lieferperiode des Produkts [cite: 235]
        - signal_time_bid/ask: Preise während der Signalgenerierung (für Slippage-Analyse).
        - execution_bid/ask: Preise während der Order-Ausführung (für PnL-Berechnung)[cite: 1176].
        - pred_<Model>: Feature-basierte Modell-Vorhersagen[cite: 787].
    """

    # 1. Multi-Stage Price Alignment
    df_backtest = align_market_prices_to_execution_window(
        df_signal_metadata,
        df_raw_lob_data,
        latency_sec=execution_latency_sec,
        staleness_tolerance_sec=price_staleness_limit_sec
    )

    # 2. Integration der Vorhersage-Vektoren
    for model_name, predictions in model_predictions.items():
        df_backtest[f"pred_{model_name}"] = predictions

    # 3. Konsistenzprüfung: Nur Snapshots mit vollständiger Preiskette behalten
    # Wir prüfen hier sowohl Signal- als auch Ausführungspreise
    required_prices = ['signal_time_bid', 'signal_time_ask', 'exec_bid', 'exec_ask']
    is_fully_tradable = df_backtest[required_prices].notna().all(axis=1)

    if is_fully_tradable.sum() < len(df_backtest):
        dropped = len(df_backtest) - is_fully_tradable.sum()
        print(f"ℹ️ Prepared {is_fully_tradable.sum():,} samples. Dropped {dropped:,} due to LOB gaps.")

    return df_backtest[is_fully_tradable].copy().sort_values(['delivery_start', 'snapshot_times'])


def calculate_dynamic_thresholds(y_pred: np.ndarray, n_std_entry: float, n_std_exit: float = 1) -> tuple:
    """Berechnet symmetrische Signalschwellen basierend auf Vorhersage-Volatilität."""
    median = np.nanmedian(y_pred)
    std = np.nanstd(y_pred)
    return (
        median + (n_std_entry * std),  # t_long_entry
        median - (n_std_entry * std),  # t_short_entry
        median + (n_std_exit * std),  # t_long_exit
        median - (n_std_exit * std)  # t_short_exit
    )


def run_economic_performance_sweep(
        df_backtest_substrate: pd.DataFrame,
        trading_strategy_logic: Callable,
        selectivity_range_n_std: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
        t_exit_ratio: float = -1,
        **strategy_hyperparameters
) -> pd.DataFrame:
    """
    Führt die ökonomische Evaluation (Kapitel 5.3) auf einem vorkalkulierten
    Backtest-Substrat aus.

    Diese Funktion isoliert die Performance-Metriken für verschiedene
    Selektivitätsstufen und Modellarchitekturen, um die Forschungsfragen
    RQ2 (Economic Utility) und RQ4 (Complexity Trade-off) zu beantworten.
    """

    # Ermittlung der zeitlichen Basis für die Trade-Frequenz-Normierung
    # Nutzt die bereits im Substrat enthaltenen Snapshot-Zeitstempel
    unique_trading_days = len(df_backtest_substrate['snapshot_times'].dt.date.unique())

    # Extraktion der verfügbaren Modelle (identifizierbar am Präfix 'pred_')
    # Erstellt auf Basis der in Kapitel 4.1 & 4.2 definierten Architekturen
    active_models = [c.replace('pred_', '') for c in df_backtest_substrate.columns if c.startswith('pred_')]

    performance_registry = []

    for model_name in tqdm(active_models, desc="Economic Performance Sweep"):
        model_column = f"pred_{model_name}"
        prediction_series = df_backtest_substrate[model_column].values

        for n_std in selectivity_range_n_std:

            # Berechnung dynamischer Signalschwellen (Kapitel 4.3.3)
            # t_long/short definieren die Selektivität des Modells
            t_long_entry, t_short_entry, t_long_exit, t_short_exit, = calculate_dynamic_thresholds(
                prediction_series,
                n_std, n_std * t_exit_ratio
            )

            # --- STRATEGIE-SIMULATION ---
            # Injizierte Handelslogik generiert den Trade-Log
            simulated_trades = trading_strategy_logic(
                df_backtest_substrate=df_backtest_substrate,
                model_column=model_column,
                t_long=t_long_entry,
                t_short=t_short_entry,
                t_exit_long=t_long_exit,
                t_exit_short=t_short_exit,
                **strategy_hyperparameters
            )

            if simulated_trades:
                df_trade_log = pd.DataFrame(simulated_trades)

                # Aggregation der ökonomischen Kennzahlen (Kapitel 4.3.2)
                # Fokus auf Sharpe Ratio, MDD und VaR
                economic_metrics = TradingEvaluator.calculate_all(
                    df_trade_log,
                    unique_trading_days
                )
            else:
                # Fallback für Phasen ohne Signale (hohe Selektivität)
                economic_metrics = {
                    "Annualized Sharpe Ratio": 0.0,
                    "Total PnL [EUR]": 0.0,
                    "Total Trades": 0
                }

            # Metadaten-Update für das spätere Dashboard-Plotting
            economic_metrics.update({
                'Model': model_name,
                'n_std': n_std,
                'Strategy_Type': trading_strategy_logic.__name__
            })

            performance_registry.append(economic_metrics)

    return pd.DataFrame(performance_registry)


def plot_sweep_results_full(
        df_sweep_results: pd.DataFrame,
        n_cols: int = 2,
        save_name: str = None,
        dashboard_title: str = "Economic Performance Sweep Analysis"  # Neuer Parameter
):
    """
    Erstellt ein dynamisches Grid-Dashboard für alle im Sweep enthaltenen Metriken.
    Visualisiert die Sensitivität der ökonomischen Outcomes gegenüber dem
    Selektivitätsparameter n_std gemäß Kapitel 4.3.
    """
    # 0. Setup Styling & Model Info
    try:
        from src.eval.evaluation.style import get_model_style, apply_shap_style
        apply_shap_style()
    except ImportError:
        sns.set_theme(style="whitegrid")

    model_names = df_sweep_results['Model'].unique()

    try:
        palette, markers = get_model_style(model_names)
    except:
        palette = "husl"
        markers = True

    # 1. Metriken identifizieren
    exclude_cols = ['Model', 'n_std', 'Threshold_Long', 'Threshold_Short', 'Strategy_Type']
    metrics = [c for c in df_sweep_results.columns if c not in exclude_cols]

    num_plots = len(metrics)
    n_rows = math.ceil(num_plots / n_cols)

    # 2. Figure Initialisierung
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False
    )
    axes_flat = axes.flatten()

    # --- NEU: Globaler Titel ---
    fig.suptitle(dashboard_title, fontsize=18, fontweight='bold', y=0.95)

    # 3. Plotting Loop
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]

        sns.lineplot(
            data=df_sweep_results,
            x='n_std',
            y=metric,
            hue='Model',
            style='Model',
            palette=palette,
            markers=markers,
            dashes=False,
            markersize=8,
            linewidth=2,
            ax=ax
        )

        clean_title = metric.replace('_', ' ').title()
        ax.set_title(clean_title, fontsize=14, fontweight='bold', pad=10)

        ax.set_ylabel("Metric Value", fontsize=10)
        ax.set_xlabel(r"Selectivity Parameter ($n_{std}$)", fontsize=10)

        if any(x in metric.lower() for x in ['pnl', 'ratio', 'bps', 'sharpe', 'var']):
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        ax.get_legend().remove()
        ax.grid(True, alpha=0.3)

    # Leere Subplots entfernen
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # 4. Zentrale Legende
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=min(len(model_names), 4),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=12
    )

    # Tight Layout Anpassung: rect=[left, bottom, right, top]
    # top=0.95 lässt Platz für den suptitle
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 5. Export & Show
    if save_name:
        save_plot_folder = "figures/econ"
        save_plot(fig, save_name, save_plot_folder)

    plt.show()


def plot_sweep_results_curated(df, save_name: str = None):
    """
    Erzeugt das finale Layout für Section 5.3.1 (Fixed TypeError).
    Layout: 4x2 Grid.
    """

    # --- 1. SETUP ---
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))

    # Konsistente Farben & Modelle

    models = sorted(df['Model'].unique())
    palette, markers = get_model_style(models)

    # Standard Plot-Argumente (dashes=False ist Standard für Solid Lines)
    kwargs = {
        'x': 'n_std',
        'hue': 'Model',
        'hue_order': models,
        'palette': palette,
        'linewidth': 1.5,
        'markers': markers,
        'markersize': 6,
        'dashes': False
    }

    # =========================================================================
    # ZEILE 1: MAIN KPI (Sharpe & Total PnL)
    # =========================================================================

    # PANEL A: Sharpe Ratio
    ax = axes[0, 0]
    sns.lineplot(data=df, y='Annualized Sharpe Ratio', ax=ax, style='Model', **kwargs)
    ax.set_title("A: Annualized Sharpe Ratio", loc='left', fontweight='bold')
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_xlabel("")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # PANEL B: Total PnL
    ax = axes[0, 1]
    sns.lineplot(data=df, y='Total PnL [EUR]', ax=ax, style='Model', **kwargs)
    ax.set_title("B: Total PnL", loc='left', fontweight='bold')
    ax.set_ylabel("Net PnL [EUR]")
    ax.set_xlabel("")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # =========================================================================
    # ZEILE 2: AVERAGE TRADE QUALITY
    # =========================================================================

    # PANEL C: Mean PnL per Trade
    ax = axes[1, 0]
    sns.lineplot(data=df, y='Mean PnL per Trade [EUR]', ax=ax, style='Model', **kwargs)
    ax.set_title("C: Mean PnL per Trade", loc='left', fontweight='bold')
    ax.set_ylabel("Avg. PnL [EUR]")
    ax.set_xlabel("")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # PANEL D: Trimmed PnL
    ax = axes[1, 1]
    sns.lineplot(data=df, y='Trimmed Pnl (+-0.5%) per Trade [EUR]', ax=ax, style='Model', **kwargs)
    ax.set_title("D: Trimmed Mean PnL per Trade (excl. 1% outliers)", loc='left', fontweight='bold')
    ax.set_ylabel("Avg. Trimmed PnL [EUR]")
    ax.set_xlabel("")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # =========================================================================
    # ZEILE 3: DIRECTIONAL ANALYSIS
    # =========================================================================

    # PANEL E: Long PnL
    ax = axes[2, 0]
    sns.lineplot(data=df, y='Long PnL [EUR]', ax=ax, style='Model', **kwargs)
    ax.set_title("E: Long Side Total PnL", loc='left', fontweight='bold')
    ax.set_ylabel("Total Long PnL [EUR]")
    ax.set_xlabel("")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # PANEL F: Short PnL
    ax = axes[2, 1]
    sns.lineplot(data=df, y='Short PnL [EUR]', ax=ax, style='Model', **kwargs)
    ax.set_title("F: Short Side Total PnL", loc='left', fontweight='bold')
    ax.set_ylabel("Total Short PnL [EUR]")
    ax.set_xlabel("")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # =========================================================================
    # ZEILE 4: RISK & TAIL PROFILE (FIXED)
    # =========================================================================

    # PANEL G: Max Drawdown
    ax = axes[3, 0]
    sns.lineplot(data=df, y='Max Drawdown [EUR]', ax=ax, style='Model', **kwargs)
    ax.set_title("G: Maximum Drawdown", loc='left', fontweight='bold')
    ax.set_ylabel("Max Drawdown [EUR]")
    ax.set_xlabel(r"Selectivity Threshold ($n_{std}$)")
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.get_legend().remove()

    # PANEL H: Tail Profile (Upside vs VaR)
    ax = axes[3, 1]

    upside_col = 'Upside Potential (0.95) [EUR]'
    var_col = 'VaR (0.05) [EUR]'

    # Layer 1: Upside (Solid Line)
    # Hier nutzen wir **kwargs ganz normal (enthält dashes=False)
    if upside_col in df.columns:
        sns.lineplot(
            data=df, y=upside_col, ax=ax, style='Model', alpha=0.9, **kwargs
        )

    # Layer 2: VaR (Dashed Line)
    if var_col in df.columns:
        # KOPIE von kwargs erstellen und 'dashes' entfernen, um Konflikt zu vermeiden
        kwargs_var = kwargs.copy()
        if 'dashes' in kwargs_var:
            del kwargs_var['dashes']

        # Custom Dashes definieren (z.B. (3, 2) Pattern)
        # Seaborn erwartet eine Liste von Dash-Pattern passend zur Anzahl der Hues (Modelle)
        custom_dashes = [(2, 2)] * len(models)

        sns.lineplot(
            data=df, y=var_col, ax=ax, style='Model',
            dashes=custom_dashes,  # Hier explizit übergeben
            alpha=0.9,
            **kwargs_var  # Restliche Argumente (ohne 'dashes')
        )

    ax.set_title("H: Percentile PnL per Trade (5%, 95%)", loc='left', fontweight='bold')
    ax.set_ylabel("PnL Magnitude [EUR]")
    ax.set_xlabel(r"Selectivity Threshold ($n_{std}$)")
    ax.axhline(0, color='black', lw=1, alpha=0.5)

    # Manuelle Legende für Metriken (Solid/Dashed)
    if ax.get_legend(): ax.get_legend().remove()

    line_solid = mlines.Line2D([], [], color='gray', linestyle='-', label='Upside (95%)')
    line_dashed = mlines.Line2D([], [], color='gray', linestyle='--', label='VaR (5%)')
    ax.legend(handles=[line_solid, line_dashed], loc='lower left', frameon=True, fontsize=8, title="Metric")

    # =========================================================================
    # GLOBALE LEGENDE (UNTEN)
    # =========================================================================

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(models),
        frameon=True,
        title_fontsize=11,
        fontsize=10
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)

    if save_name:
        save_plot_folder = "figures/econ"
        save_plot(fig, save_name, save_plot_folder)

    plt.show()


def plot_trading_performance_deep_dive(
        trade_log: pd.DataFrame,
        model_name: str,
        n_std_used: float,
        dashboard_title: str = "Trading Strategy Deep Dive"
):
    """
    Erstellt ein umfangreiches Performance-Dashboard.
    Panel B zeigt nun die kumulierten Reibungsverluste (Slippage + Spread).
    """
    if trade_log is None or trade_log.empty:
        print(f"⚠️ Hinweis: Trade Log für {model_name} ist leer.")
        return

    # --- DATENAUFBEREITUNG ---
    df = trade_log.copy()
    for col in ['exit_time', 'entry_time', 'delivery_start']:
        df[col] = pd.to_datetime(df[col])

    df = df.sort_values('exit_time')

    # 1. Zeitreihen-Vorbereitung (Aggregation simultaner Exits)
    full_time_idx = df['exit_time'].unique()

    # Aggregierte PnLs pro Zeitstempel
    total_daily = df.groupby('exit_time')['pnl'].sum().reindex(full_time_idx).fillna(0).cumsum()
    long_daily = df[df['side'] == 'LONG'].groupby('exit_time')['pnl'].sum().reindex(full_time_idx).fillna(0).cumsum()
    short_daily = df[df['side'] == 'SHORT'].groupby('exit_time')['pnl'].sum().reindex(full_time_idx).fillna(0).cumsum()

    # 2. Risiko-Metriken (auf Gesamt-Equity)
    high_watermark = total_daily.cummax()
    drawdown = total_daily - high_watermark

    # 3. Zeit- & Lifecycle-Features
    df['delivery_hour'] = df['delivery_start'].dt.hour
    df['ttd_at_entry'] = (df['delivery_start'] - df['entry_time']).dt.total_seconds() / 60

    ttd_bins = range(0, 301, 15)
    ttd_labels = [f"{i}-{i + 15}" for i in ttd_bins[:-1]]
    df['ttd_cluster'] = pd.cut(df['ttd_at_entry'], bins=ttd_bins, labels=ttd_labels)

    # --- VISUALISIERUNG ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1, 1, 1.2])

    fig.suptitle(f"{dashboard_title}\nModel: {model_name} | Selectivity: $n_{{std}}={n_std_used}$",
                 fontsize=18, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(gs[0, :])  # A: Kombinierte Equity
    ax2 = fig.add_subplot(gs[1, :])  # B: Frictional Losses (Slippage & Spread)
    ax3 = fig.add_subplot(gs[2, 0])  # C: Drawdown
    ax4 = fig.add_subplot(gs[2, 1])  # D: PnL Histogramm
    ax5 = fig.add_subplot(gs[3, 0])  # E: Hourly
    ax6 = fig.add_subplot(gs[3, 1])  # F: Lifecycle

    # --- A: INTEGRATED EQUITY CURVE ---
    ax1.plot(full_time_idx, total_daily, color='black', linewidth=3, label='Total Net PnL', zorder=5)
    ax1.fill_between(full_time_idx, 0, total_daily, color='gray', alpha=0.1, zorder=1)
    ax1.plot(full_time_idx, long_daily, color='#1f77b4', linewidth=1.5, label='Long PnL', alpha=0.8, zorder=4)
    ax1.plot(full_time_idx, short_daily, color='#d62728', linewidth=1.5, label='Short PnL', alpha=0.8, zorder=3)

    ax1.set_title("A: Integrated Equity Curve (Directional Breakdown)", fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylabel("Cumulative PnL [EUR]")
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)

    # --- B: FRICTIONAL LOSSES (Slippage & Spread) ---
    # Wir stacken die Verluste, um die Gesamtreibung zu zeigen
    if 'slippage_loss' in df.columns and 'spread_loss' in df.columns:
        cum_slippage = df.groupby('exit_time')['slippage_loss'].sum().reindex(full_time_idx).fillna(0).cumsum()
        cum_spread = df.groupby('exit_time')['spread_loss'].sum().reindex(full_time_idx).fillna(0).cumsum()
        total_friction = cum_slippage + cum_spread

        # Plotten der einzelnen Layer
        ax2.plot(full_time_idx, total_friction, color='firebrick', linewidth=2,
                 label='Total Friction (Slippage + Spread)')
        ax2.fill_between(full_time_idx, cum_spread, total_friction, color='red', alpha=0.2,
                         label='Alpha Erosion (Latenz)')
        ax2.fill_between(full_time_idx, 0, cum_spread, color='orange', alpha=0.2, label='Liquidity Cost (Spread)')

        ax2.set_title("B: Frictional Decomposition: Alpha Erosion vs. Liquidity Costs", fontsize=14, fontweight='bold',
                      loc='left')
        ax2.set_ylabel("Lost Alpha [EUR]")
        ax2.legend(loc='upper left')

    # --- C: DRAWDOWN RISK ---
    ax3.fill_between(full_time_idx, drawdown, 0, color='#d62728', alpha=0.3)
    ax3.plot(full_time_idx, drawdown, color='#d62728', linewidth=1)
    ax3.set_title("C: Portfolio Drawdown", fontsize=14, fontweight='bold', loc='left')
    ax3.set_ylabel("Drawdown [EUR]")

    # --- D: PNL PER TRADE DISTRIBUTION ---
    sns.histplot(df['pnl'], kde=True, ax=ax4, color='purple', bins=100)
    ax4.axvline(df['pnl'].mean(), color='black', linestyle='--', label=f"Mean: {df['pnl'].mean():.2f}")
    ax4.set_title("D: Trade Return Distribution", fontsize=14, fontweight='bold', loc='left')
    ax4.legend()

    # --- E: HOURLY PROFITABILITY ---
    sns.boxplot(data=df, x='delivery_hour', y='pnl', ax=ax5, palette="Blues", showfliers=False)
    ax5.set_title("E: PnL by Delivery Hour", fontsize=14, fontweight='bold', loc='left')
    ax5.axhline(0, color='red', linestyle='--', alpha=0.6)

    # --- F: LIFECYCLE (TTD) ---
    ordered_ttd = ttd_labels[::-1]
    sns.boxplot(data=df, x='ttd_cluster', y='pnl', order=ordered_ttd, ax=ax6, palette="Reds", showfliers=False)
    ax6.set_title("F: PnL by Time-To-Delivery (Lifecycle)", fontsize=14, fontweight='bold', loc='left')
    plt.xticks(rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_local_extrema_indices(series, distance=14):
    """
    Findet lokale Maxima und Minima in einer Zeitreihe.
    distance: Mindestabstand in Tagen zwischen zwei Markern (Glättung).
    """
    y = series.values
    # 1. Lokale Maxima (Peaks)
    peaks, _ = find_peaks(y, distance=distance)

    # 2. Lokale Minima (Invertierte Peaks)
    valleys, _ = find_peaks(-y, distance=distance)

    # Zusammenfügen und sortieren
    indices = np.concatenate((peaks, valleys))
    indices.sort()
    return indices


def plot_regime_impact(
        df_backtest_substrate: pd.DataFrame,
        model_thresholds: dict = None,  # NEU: Dict für spezifische Thresholds, z.B. {'MLP': 2.0, 'CatBoost': 2.5}
        default_n_std: float = 2.0,  # NEU: Fallback, falls ein Modell nicht im Dict steht
        target_t_exit_ratio: float = -1.0,
        spread_factor: float = 0.5,
        models_to_compare: list = None,
        linthresh: float = 400.0,
        rolling_days=7,
        save_name=None,
        # Parameter für X_test Features
        X_test: pd.DataFrame = None,
        feature_cols: list = None,
        feature_titles: list = None,
        feature_y_labels: list = None
):
    """
    Visualisierung von Equity, Daily PnL und Markt-Regimes.
    Features aus X_test werden über 'snapshot_times' auf Tagesbasis aggregiert
    und mit einem daily rolling window geglättet.
    Individuelle Thresholds pro Modell werden unterstützt.
    """

    if model_thresholds is None:
        model_thresholds = {}

    # --- 1. SETUP & ZEITACHSEN ---
    if models_to_compare is None:
        models_to_compare = [c.replace('pred_', '') for c in df_backtest_substrate.columns if c.startswith('pred_')]
        sort_order = ['LASSO', 'EBM', 'RF', 'CatBoost', 'MLP']
        models_to_compare = sorted(models_to_compare, key=lambda x: sort_order.index(x) if x in sort_order else 99)

    events = {
        "IDCC": (pd.Timestamp("2024-05-22"), "#d62728"),
        "IDA": (pd.Timestamp("2024-06-13"), "#ff7f0e")
    }

    full_time_idx = pd.date_range(
        start=df_backtest_substrate['delivery_start'].min(),
        end=df_backtest_substrate['delivery_start'].max(),
        freq='h'
    )
    # Referenz-Index für tägliche Daten
    daily_idx = pd.date_range(full_time_idx.min().floor('D'), full_time_idx.max().floor('D'), freq='D')

    equity_curves = {}
    daily_pnl_curves = {}

    # --- 2. MODELL-DATEN GENERIEREN ---
    for model_name in tqdm(models_to_compare, desc="Berechne Modelle"):
        model_col = f"pred_{model_name}"

        # Hole den spezifischen Threshold für das Modell (oder Fallback)
        current_n_std = model_thresholds.get(model_name, default_n_std)

        t_long, t_short, t_l_exit, t_s_exit = calculate_dynamic_thresholds(
            df_backtest_substrate[model_col], current_n_std, current_n_std * target_t_exit_ratio
        )
        trades = TradingStrategyManager.run_threshold_exit_strategy(
            df_backtest_substrate, model_col, t_long, t_short, t_l_exit, t_s_exit, spread_factor=spread_factor
        )

        if not trades: continue
        df_tr = pd.DataFrame(trades)
        df_tr['exit_time'] = pd.to_datetime(df_tr['exit_time'])

        # Panel A: Hourly Equity
        hourly_pnl = df_tr.groupby(df_tr['exit_time'].dt.floor('h'))['pnl'].sum()
        equity_curves[model_name] = hourly_pnl.reindex(full_time_idx).fillna(0).cumsum()

        # Panel B: Daily Rolling PnL
        daily_sum = df_tr.groupby(df_tr['exit_time'].dt.floor('D'))['pnl'].sum()
        daily_pnl_curves[model_name] = daily_sum.reindex(daily_idx).fillna(0).rolling(window=rolling_days,
                                                                                      center=True).mean()

    # --- 3. SUBPLOTS SETUP ---
    num_features = len(feature_cols) if feature_cols is not None else 0
    total_subplots = 2 + num_features
    height_ratios = [1.2, 1.2] + [0.5] * num_features

    fig, axes = plt.subplots(total_subplots, 1, figsize=(12, 5 + 2 * num_features),
                             sharex=True, gridspec_kw={'height_ratios': height_ratios})

    # Wenn nur ein Subplot da wäre (was bei 2 + x eigentlich nicht passiert, aber sicherheitshalber)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    ax1, ax2 = axes[0], axes[1]
    colors, markers = get_model_style(models_to_compare)

    # --- PANEL A: EQUITY ---
    for model in models_to_compare:
        if model in equity_curves:
            clean_series = equity_curves[model].fillna(0)
            # distance=5 bedeutet: Mindestens 5 Tage Abstand zwischen Markern
            peak_indices = get_local_extrema_indices(clean_series, distance=600)

            # Inkludiere den Threshold im Label für die Legende
            current_n_std = model_thresholds.get(model, default_n_std)
            label_str = f"{model} ($n_{{std}}={current_n_std}$)"

            ax1.plot(equity_curves[model].index, equity_curves[model], label=label_str, color=colors[model],
                     linewidth=1.5,
                     alpha=0.9, marker=markers[model], markevery=peak_indices, ms=6, mew=0.3)

    ax1.set_title(f"A: Total Net PnL", fontsize=12, fontweight='bold', loc='left')
    ax1.set_ylabel("Total Net PnL [EUR]")
    ax1.axhline(0, color='black', lw=1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9, frameon=True, ncols=2)

    # --- PANEL B: DAILY PNL (SYMLOG) ---
    df_daily_all = pd.DataFrame(daily_pnl_curves)
    g_min, g_max = df_daily_all.min().min(), df_daily_all.max().max()

    for model in models_to_compare:
        if model in daily_pnl_curves:
            # >>> PEAKS BERECHNEN <<<
            clean_series = daily_pnl_curves[model].fillna(0)
            peak_indices = get_local_extrema_indices(clean_series, distance=7)

            # Label analog zu Panel A (Threshold anzeigen)
            current_n_std = model_thresholds.get(model, default_n_std)
            label_str = f"{model} ($n_{{std}}={current_n_std}$)"

            ax2.plot(daily_pnl_curves[model].index, daily_pnl_curves[model], label=label_str, color=colors[model],
                     linewidth=1.5, alpha=0.9,
                     marker=markers[model], markevery=peak_indices, ms=6, mew=0.3)

    ax2.set_yscale('symlog', linthresh=linthresh)
    ax2.yaxis.set_major_locator(
        FixedLocator(np.unique(np.concatenate([np.arange(-int(linthresh), int(linthresh) + 1, 100), [g_min, g_max]]))))
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_title(f"B: Daily PnL ({rolling_days}d Rolling Mean)", fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylabel("Daily PnL [EUR]")
    ax2.axhline(0, color='black', lw=1)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9, frameon=True, ncols=2)

    # --- NEUE PANELS: DAILY ROLLING FEATURES ---
    if num_features > 0 and X_test is not None:
        # Sicherstellen, dass snapshot_times datetime ist
        X_tmp = X_test.copy()
        X_tmp['snapshot_day'] = pd.to_datetime(X_tmp['snapshot_times']).dt.floor('D')

        for i, col in enumerate(feature_cols):
            cur_ax = axes[2 + i]

            # 1. Daily Aggregation (Mittelwert pro Tag)
            daily_feat = X_tmp.groupby('snapshot_day')[col].mean()

            # 2. Daily Rolling
            feat_rolled = daily_feat.reindex(daily_idx).fillna(method='ffill').rolling(window=rolling_days,
                                                                                       center=True).mean()

            ylim_max = feat_rolled.max() * 1.1
            ylim_min = feat_rolled.min() * 0.9

            # Plot
            cur_ax.plot(feat_rolled.index, feat_rolled, color='#444444', linewidth=1.3, label=col)
            cur_ax.fill_between(feat_rolled.index, feat_rolled, color='gray', alpha=0.15)

            # Labels
            title = feature_titles[i] if (feature_titles and i < len(feature_titles)) else f"Feature: {col}"
            y_lab = feature_y_labels[i] if (feature_y_labels and i < len(feature_y_labels)) else "Daily Avg (Smoothed)"

            cur_ax.set_ylim(ylim_min, ylim_max)

            cur_ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
            cur_ax.set_ylabel(y_lab)
            cur_ax.grid(True, alpha=0.3)
            cur_ax.legend(loc='upper left', fontsize=9, frameon=True)

    # --- FORMATIERUNG & EVENTS ---
    for ax in axes:
        for event_name, event_data in events.items():
            min_time, max_time = ax.get_xlim()

            # Wandle das Event-Datum in das Matplotlib-Zahlenformat um (Tage statt Sekunden)
            event_num = mdates.date2num(event_data[0])

            if min_time <= event_num <= max_time:
                ax.axvline(event_data[0], color=event_data[1], linestyle='--', alpha=0.7)

                if ax == axes[0]:
                    ax.text(event_data[0], ax.get_ylim()[1] * 0.95, event_name, color=event_data[1],
                            fontsize=9, fontweight='bold', rotation=90, va='top', ha='right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # wenn mehr als 2 monate im plot sind, dann monatliche ticks, sonst wöchentliche
    if (full_time_idx.max() - full_time_idx.min()).days > 60:
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save_name:
        save_plot_folder = "figures/econ"
        save_plot(fig, save_name, save_plot_folder)
    plt.show()


def plot_alpha_sources(
        df_backtest_substrate: pd.DataFrame,
        model_thresholds: dict = None,  # NEU: Dict für spezifische Thresholds, z.B. {'MLP': 2.0, 'CatBoost': 2.5}
        default_n_std: float = 2.0,  # NEU: Fallback, falls ein Modell nicht im Dict steht
        target_t_exit_ratio: float = -1.0,
        spread_factor: float = 0.5,
        models_to_compare: list = None,
        a_y_lim_factor: float = 1.0,
        b_c_y_lim_factor: float = 1.5,
        save_name: str = None
):
    """
    Matrix-Layout mit Bar Charts + Error Bars (Whiskers) und Histogrammen.
    - Zeile 1: Lifecycle (Area Chart)
    - Zeile 2 & 3: Bar Charts mit 95% Confidence Interval Whiskers.
    - Zeile 4: Histogramm der Trade PnLs.
    Individuelle Thresholds pro Modell werden unterstützt.
    """

    if model_thresholds is None:
        model_thresholds = {}

    # --- 1. DATENGENERIERUNG ---
    if models_to_compare is None:
        models_to_compare = [c.replace('pred_', '') for c in df_backtest_substrate.columns if c.startswith('pred_')]
        sort_order = ['LASSO', 'EBM', 'RF', 'CatBoost', 'MLP']
        models_to_compare = sorted(models_to_compare, key=lambda x: sort_order.index(x) if x in sort_order else 99)

    comparison_data = []
    print("Generiere Profile mit modellspezifischen Thresholds...")

    for model_name in tqdm(models_to_compare):
        model_col = f"pred_{model_name}"

        # Hole den spezifischen Threshold für das Modell (oder Fallback)
        current_n_std = model_thresholds.get(model_name, default_n_std)

        t_long, t_short, t_l_exit, t_s_exit = calculate_dynamic_thresholds(
            df_backtest_substrate[model_col], current_n_std, current_n_std * target_t_exit_ratio
        )
        trades = TradingStrategyManager.run_threshold_exit_strategy(
            df_backtest_substrate, model_col, t_long, t_short, t_l_exit, t_s_exit, spread_factor=spread_factor
        )

        if trades:
            df = pd.DataFrame(trades)
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['delivery_start'] = pd.to_datetime(df['delivery_start'])

            df['ttd_min'] = (df['delivery_start'] - df['entry_time']).dt.total_seconds() / 60
            df['ttd_cluster'] = (df['ttd_min'] // 10) * 10

            df['trading_hour'] = df['entry_time'].dt.hour
            df['delivery_hour'] = df['delivery_start'].dt.hour
            df['Model'] = model_name

            low = df['pnl'].quantile(0.005)
            high = df['pnl'].quantile(0.995)
            df = df[(df['pnl'] >= low) & (df['pnl'] <= high)]

            comparison_data.append(df)

    if not comparison_data:
        print("Keine Trades gefunden.")
        return

    df_all = pd.concat(comparison_data, ignore_index=True)

    # Aggregation nur für Zeile 1 (Mean Verlauf)
    df_life = df_all.groupby(['Model', 'ttd_cluster'])['pnl'].mean().reset_index()
    df_life = df_life.sort_values(by=['Model', 'ttd_cluster'], ascending=[True, False])

    # Y-Limits berechnen (basierend auf den Means + etwas Puffer für die Whiskers)
    ylim_a = max(abs(df_life['pnl'].min()), abs(df_life['pnl'].max())) * a_y_lim_factor
    max_mean_b = df_all.groupby(['Model', 'trading_hour'])['pnl'].mean().abs().max()
    ylim_b = max_mean_b * b_c_y_lim_factor
    max_mean_c = df_all.groupby(['Model', 'delivery_hour'])['pnl'].mean().abs().max()
    ylim_c = max_mean_c * b_c_y_lim_factor

    # Für Histogramme: Globale X- und Y-Limits berechnen, damit sie vergleichbar sind
    pnl_min = df_all['pnl'].min()
    pnl_max = df_all['pnl'].max()
    hist_bins = np.linspace(pnl_min, pnl_max, 40)  # 40 Bins über die gesamte Range

    max_hist_count = 0
    for m in models_to_compare:
        subset_pnl = df_all[df_all['Model'] == m]['pnl']
        if not subset_pnl.empty:
            counts, _ = np.histogram(subset_pnl, bins=hist_bins)
            max_hist_count = max(max_hist_count, counts.max())
    ylim_d = max_hist_count * 1.1  # 10% Puffer nach oben für die Y-Achse

    # --- 2. PLOTTING ---
    n_models = len(models_to_compare)
    # Höhe angepasst (zuvor 10, jetzt 13.5 für die zusätzliche Zeile)
    fig = plt.figure(figsize=(3 * n_models, 13.5))
    # 4 Zeilen statt 3
    gs = gridspec.GridSpec(4, n_models, height_ratios=[1, 1, 1, 1], hspace=0.45, wspace=0.05)

    colors, _ = get_model_style(models_to_compare)

    # Grid Definitionen
    ticks_life_major = np.arange(0, 301, 60)
    ticks_life_minor = np.arange(0, 301, 30)
    ticks_hours_major = [0, 6, 12, 18, 23]
    ticks_hours_minor = np.arange(0, 24, 1)

    for i, model in enumerate(models_to_compare):
        color = colors.get(model, 'gray')
        current_n_std = model_thresholds.get(model, default_n_std)

        model_subset = df_all[df_all['Model'] == model]

        # === ROW 1: LIFECYCLE (Area Chart - Mean) ===
        ax1 = fig.add_subplot(gs[0, i])
        data_life = df_life[df_life['Model'] == model]

        ax1.plot(data_life['ttd_cluster'], data_life['pnl'], color=color, lw=1.5)
        ax1.fill_between(data_life['ttd_cluster'], data_life['pnl'], 0, color=color, alpha=0.2)

        ax1.set_xlim(300, 0)
        ax1.set_ylim(-ylim_a, ylim_a)

        # Grid Row 1
        ax1.set_xticks(ticks_life_major)
        ax1.set_xticks(ticks_life_minor, minor=True)
        ax1.grid(True, which='major', axis='x', linestyle=':', alpha=0.7, color='gray')
        ax1.grid(True, which='minor', axis='x', linestyle=':', alpha=0.3, color='gray')
        ax1.grid(True, axis='y', alpha=0.2)
        ax1.axhline(0, color='black', lw=0.8)

        ax1.axvline(60, color='gray', ls='-', lw=1.5, alpha=0.3)
        ax1.axvline(30, color='orange', ls='-', lw=1.5, alpha=0.3)

        # Titel zeigt nun Modell und individuellen Threshold
        ax1.set_title(f"{model}\n($n_{{std}}={current_n_std}$)", fontweight='bold', fontsize=12)

        if i == 0:
            ax1.set_ylabel("Mean PnL [EUR]\n(Lifecycle)", fontweight='bold')
            ax1.text(60, ylim_a * 0.97, "SIDC GC (T-60)", color='gray', rotation=90, fontsize=8, ha='right',
                     va="top")
            ax1.text(30, ylim_a * 0.97, "CZ-GC (T-30)", color='orange', rotation=90, fontsize=8, ha='right',
                     va="top")
        else:
            ax1.set_yticklabels([])
            ax1.set_ylabel("")

        ax1.set_xticklabels([str(t) for t in ticks_life_major], fontsize=9)
        if i == n_models // 2: ax1.set_xlabel("Minutes to Delivery")

        # === ROW 2: TRADING HOUR (Barplot + Whiskers) ===
        ax2 = fig.add_subplot(gs[1, i])

        sns.barplot(
            data=model_subset, x='trading_hour', y='pnl', ax=ax2,
            color=color, alpha=0.8, edgecolor=None,
            errorbar=('ci', 95),
            capsize=0.15,
            err_kws={'linewidth': 1, 'color': 'black'}
        )

        ax2.set_ylim(-ylim_b, ylim_b)
        ax2.axhline(0, color='black', lw=0.8)

        # Grid Row 2
        ax2.xaxis.set_major_locator(FixedLocator(ticks_hours_major))
        ax2.xaxis.set_minor_locator(FixedLocator(ticks_hours_minor))
        ax2.grid(True, which='major', axis='x', linestyle='-', alpha=0.5, color='gray')
        ax2.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2, color='gray')
        ax2.grid(True, axis='y', alpha=0.2)

        ax2.set_xticklabels([str(t) for t in ticks_hours_major])

        if i == 0:
            ax2.set_ylabel("Mean PnL [EUR]\n(Hour of Day)", fontweight='bold')
        else:
            ax2.set_yticklabels([])
            ax2.set_ylabel("")

        ax2.set_xlabel("")
        if i == n_models // 2: ax2.set_xlabel("Hour of Day")

        # === ROW 3: DELIVERY HOUR (Barplot + Whiskers) ===
        ax3 = fig.add_subplot(gs[2, i])

        sns.barplot(
            data=model_subset, x='delivery_hour', y='pnl', ax=ax3,
            color=color, alpha=0.8, edgecolor=None,
            errorbar=('ci', 95),
            capsize=0.15,
            err_kws={'linewidth': 1, 'color': 'black'}
        )

        ax3.set_ylim(-ylim_c, ylim_c)
        ax3.axhline(0, color='black', lw=0.8)

        # Grid Row 3
        ax3.xaxis.set_major_locator(FixedLocator(ticks_hours_major))
        ax3.xaxis.set_minor_locator(FixedLocator(ticks_hours_minor))
        ax3.grid(True, which='major', axis='x', linestyle='-', alpha=0.5, color='gray')
        ax3.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2, color='gray')
        ax3.grid(True, axis='y', alpha=0.2)

        ax3.set_xticklabels([str(t) for t in ticks_hours_major])

        if i == 0:
            ax3.set_ylabel("Mean PnL [EUR]\n(Product)", fontweight='bold')
        else:
            ax3.set_yticklabels([])
            ax3.set_ylabel("")

        ax3.set_xlabel("")
        if i == n_models // 2: ax3.set_xlabel("Delivery Hour")

        # === ROW 4: PnL HISTOGRAM (Verteilung - Log) ===
        ax4 = fig.add_subplot(gs[3, i])

        sns.histplot(
            data=model_subset, x='pnl', bins=hist_bins, ax=ax4,
            color=color, alpha=0.8, edgecolor='white', linewidth=0.5
        )

        ax4.set_yscale('log')

        # 1. WICHTIG: Y-Limit zwingend hochsetzen, sonst sieht man den 1000er Log-Tick nicht!
        current_ylim_max = max(1000, max_hist_count * 2.0)
        ax4.set_ylim(0, current_ylim_max)
        ax4.set_xlim(-pnl_max, pnl_max)

        ax4.axvline(0, color='black', lw=0.8, linestyle='--')
        if i == 0:
            ax4.set_ylabel("Frequency (Log-Scale)\n(Trade PnL)", fontweight='bold')
        else:
            ax4.set_ylabel("")
            # 4. WICHTIG: Labels sicher ausblenden, ohne die Ticks zu zerstören
            ax4.tick_params(axis='y', labelleft=False)

        ax4.grid(True, which='major', alpha=0.5)

        ax4.set_xlabel("")
        if i == n_models // 2:
            ax4.set_xlabel("Trade PnL [EUR]")

    # Overall Title angepasst
    if not model_thresholds:
        title = f"Structural Performance (Mean with 95% CI) | Uniform Thresholds"
    else:
        title = f"Structural Performance (Mean with 95% CI) | Model-Specific Thresholds"
    fig.suptitle(title, y=0.96, fontsize=16)

    if save_name:
        save_plot_folder = "figures/econ"
        save_plot(fig, save_name, save_plot_folder)

    plt.show()
