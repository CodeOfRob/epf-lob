import pandas as pd
import numpy as np
from typing import Dict, Any


class TradingEvaluator:
    """
    Berechnet ökonomische Metriken basierend auf den Definitionen in Kapitel 4.
    """

    @staticmethod
    def calculate_all(trade_log: pd.DataFrame, evaluation_days: int) -> Dict[str, Any]:
        if trade_log is None or trade_log.empty:
            return {"Total PnL [EUR]": 0.0, "Total Trades": 0}

        pnls = trade_log['pnl'].values

        # Aggregation der Metrik-Gruppen
        return {
            **TradingEvaluator._calc_risk_adjusted(trade_log, pnls),
            **TradingEvaluator._calc_profit_and_frequency(trade_log, pnls, evaluation_days),
            **TradingEvaluator._calc_directional_stats(trade_log),
            **TradingEvaluator._calc_tail_metrics(pnls),
            **TradingEvaluator._calculate_economic_efficiency(trade_log),

        }

    @staticmethod
    def _calc_profit_and_frequency(df: pd.DataFrame, pnls: np.ndarray, days: int) -> Dict[str, Any]:
        return {
            'Total PnL [EUR]': np.sum(pnls),
            'Mean PnL per Trade [EUR]': np.mean(pnls),
            'Trimmed Pnl (+-0.5%) per Trade [EUR]': np.mean(pnls[np.abs(pnls) <= np.percentile(np.abs(pnls), 99.5)]),
            'Trade Frequency [Trades/Day]': len(pnls) / days if days > 0 else 0,
            'Total Trades': len(pnls),
            'Mean Holding Time per Trade [minutes]': df['exit_time'].subtract(df[
                                                                                  'entry_time']).dt.total_seconds().mean() / 60 if 'entry_time' in df.columns and 'exit_time' in df.columns else 0
        }

    @staticmethod
    def _calc_risk_adjusted(df: pd.DataFrame, pnls: np.ndarray) -> Dict[str, Any]:
        # 1. Annualized Sharpe Ratio (Basis: Daily Returns)
        # Wir gruppieren PnLs nach Tag (Annahme: 'delivery_start' ist datetime)
        df_daily = df.copy()
        df_daily['date'] = pd.to_datetime(df_daily['delivery_start']).dt.date
        daily_returns = df_daily.groupby('date')['pnl'].sum()

        mu_daily = daily_returns.mean()
        sigma_daily = daily_returns.std()

        # Sharpe = (R_daily / sigma_daily) * sqrt(365.25)
        sharpe_ann = (mu_daily / sigma_daily * np.sqrt(365.25)) if sigma_daily > 0 else 0

        # 2. Maximum Drawdown (MDD)
        cum_pnl = np.cumsum(pnls)
        # Da wir mit absolutem PnL (EUR) arbeiten, ist MDD hier die Peak-to-Valley Differenz
        # Falls MDD als % gewünscht ist, müsste ein Startkapital definiert werden.
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - peak
        mdd = np.min(drawdowns)

        return {
            'Annualized Sharpe Ratio': sharpe_ann,
            'Max Drawdown [EUR]': mdd
        }

    @staticmethod
    def _calc_tail_metrics(pnls: np.ndarray) -> Dict[str, Any]:
        # Value at Risk (VaR 0.05)
        # Entspricht dem 5. Perzentil der PnL-Verteilung
        var_05 = np.percentile(pnls, 5) if len(pnls) > 0 else 0
        upside_05 = np.percentile(pnls, 95) if len(pnls) > 0 else 0
        return {
            'VaR (0.05) [EUR]': var_05,
            'Upside Potential (0.95) [EUR]': upside_05
        }

    @staticmethod
    def _calculate_economic_efficiency(df_trade_log: pd.DataFrame) -> dict:
        """
        Berechnet, wie viel vom theoretischen Brutto-Alpha nach Abzug von
        Latenz (Slippage) und Liquidität (Spread) übrig bleibt.
        """
        if df_trade_log.empty:
            return {"Economic Efficiency Ratio": 0.0, "Alpha Capture Ratio": 0.0}

        total_net_pnl = df_trade_log['pnl'].sum()
        total_slippage = df_trade_log['slippage_loss'].sum()
        total_spread = df_trade_log['spread_loss'].sum()

        # Brutto-Alpha = Netto + Reibungsverluste
        gross_alpha = total_net_pnl + total_slippage + total_spread

        # 1. Economic Efficiency: Netto vs. Alles (Wie viel landet auf dem Konto?)
        # Vermeidung von Division by Zero bei leeren/neutralen Phasen
        efficiency_ratio = total_net_pnl / gross_alpha if gross_alpha > 0 else 0.0

        # 2. Alpha Capture: Netto+Spread vs. Brutto (Wie resistent ist das Signal gegen Latenz?)
        capture_ratio = (total_net_pnl + total_spread) / gross_alpha if gross_alpha > 0 else 0.0

        return {
            "Economic Efficiency Ratio": efficiency_ratio,
            "Alpha Capture Ratio": capture_ratio,  # um grunde nur zeitabhängig
            "Total Frictional Loss [EUR]": total_slippage + total_spread
        }

    @staticmethod
    def _calc_directional_stats(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty or 'side' not in df.columns:
            return {
                "Long PnL [EUR]": 0.0,
                "Short PnL [EUR]": 0.0,
                "PnL Balance (L/S)": 0.0,  # 0 = Neutral
                "Total Trades Long": 0,
                "Total Trades Short": 0
            }

        # PnL Summen
        pnl_long = df.loc[df['side'] == 'LONG', 'pnl'].sum()
        pnl_short = df.loc[df['side'] == 'SHORT', 'pnl'].sum()

        # Absolute Summe (Total Gross Impact)
        total_abs_pnl = abs(pnl_long) + abs(pnl_short)

        # Anzahl Trades
        total_trades_long = df[df['side'] == 'LONG'].shape[0]
        total_trades_short = df[df['side'] == 'SHORT'].shape[0]

        # Balance Score:
        # > 0 bedeutet Long-Dominanz
        # < 0 bedeutet Short-Dominanz
        # Nahe 0 bedeutet ausgeglichen
        if total_abs_pnl > 0:
            balance = (pnl_long - pnl_short) / total_abs_pnl
        else:
            balance = 0.0

        return {
            "Long PnL [EUR]": pnl_long,
            "Short PnL [EUR]": pnl_short,
            "PnL Balance (L/S)": balance,
            "Total Trades Long": total_trades_long,
            "Total Trades Short": total_trades_short
        }
