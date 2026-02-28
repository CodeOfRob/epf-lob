from typing import List, Dict, Literal

import pandas as pd


class TradingStrategyManager:
    """
    Verwaltet die Ausführung von Trading-Strategien auf Basis der
    Backtest-Foundation (Kapitel 4.3.3).
    """

    @staticmethod
    def run_reversal_strategy(
            df_backtest_substrate: pd.DataFrame,
            model_column: str,
            t_long: float,
            t_short: float,
            spread_factor: float = 0.5,  # Standard gemäß Sektion 4.3.3
            trade_direction: Literal["both", "long_only", "short_only"] = 'both',
            **kwargs
    ) -> List[Dict]:
        """
        Implementiert eine Reversal-Logik: Positionen werden bei gegenteiligen
        Signalen oder am Gate Closure (T-5) geschlossen.
        """

        # Validierung der notwendigen Spalten aus build_backtest_foundation
        required = ['delivery_start', 'snapshot_times', 'exec_bid', 'exec_ask']
        if not all(col in df_backtest_substrate.columns for col in required):
            raise ValueError(f"DF fehlt notwendige Backtest-Spalten: {required}")

        trade_logs = []

        # Gruppierung nach Produkten (Hourly Contracts), um produktübergreifendes Leakage zu vermeiden [cite: 1236, 1246]
        for product_id, group in df_backtest_substrate.groupby('delivery_start', observed=True):

            # Extraktion der Vektoren für Performance
            times = group['snapshot_times'].values
            signals = group[model_column].values
            exec_bids = group['exec_bid'].values
            exec_asks = group['exec_ask'].values

            # Optionale Slippage-Daten für erweiterte Analyse (Kapitel 6.3)
            signal_bids = group['signal_time_bid'].values if 'signal_time_bid' in group else exec_bids

            current_position = 0  # 0=Flat, 1=Long, -1=Short
            entry_price = 0.0
            entry_time = None
            entry_signal_mid = 0.0

            for i in range(len(group)):
                pred = signals[i]
                bid, ask = exec_bids[i], exec_asks[i]
                time = times[i]
                is_gate_closure = (i == len(group) - 1)

                # 1. Signal-Klassifizierung gemäß Selektivität (n_std) [cite: 1219, 1220]
                raw_signal = 0
                if pred > t_long:
                    raw_signal = 1
                elif pred < t_short:
                    raw_signal = -1

                # 2. Ziel-Position bestimmen
                target_position = raw_signal if raw_signal != 0 else current_position
                if is_gate_closure:
                    target_position = 0  # Zwangsliquidierung

                # 3. Position beenden (Exit Logik)
                if current_position != 0 and target_position != current_position:
                    mid_price = (ask + bid) / 2
                    half_spread = (ask - bid) / 2

                    # Exit-Preis inkl. anteiliger Transaktionskosten [cite: 1227, 1229]
                    exit_price = mid_price - (half_spread * spread_factor) if current_position == 1 \
                        else mid_price + (half_spread * spread_factor)

                    pnl = (exit_price - entry_price) * current_position

                    # Slippage-Berechnung: Diff zwischen Mid bei Signal vs. Mid bei Execution
                    exec_mid = (ask + bid) / 2
                    slippage = exec_mid - entry_signal_mid if current_position == 1 else entry_signal_mid - exec_mid

                    trade_logs.append({
                        'delivery_start': product_id,
                        'entry_time': entry_time,
                        'exit_time': time,
                        'side': 'LONG' if current_position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'slippage_loss': slippage,
                        'reason': 'Gate Closure' if is_gate_closure else 'Signal Reversal'
                    })
                    current_position = 0

                # 4. Position eröffnen (Entry Logik)
                if current_position == 0 and target_position != 0:
                    # Filterung nach Handelsmodus (Long-Bias Test) [cite: 1623]
                    is_allowed = (target_position == 1 and trade_direction in ['both', 'long_only']) or \
                                 (target_position == -1 and trade_direction in ['both', 'short_only'])

                    if is_allowed:
                        current_position = target_position
                        entry_time = time
                        mid_price = (ask + bid) / 2
                        half_spread = (ask - bid) / 2

                        entry_signal_mid = (signal_bids[i] + (
                            group['signal_time_ask'].values[i] if 'signal_time_ask' in group else ask)) / 2

                        # Entry-Preis inkl. anteiliger Transaktionskosten [cite: 1229]
                        entry_price = mid_price + (half_spread * spread_factor) if current_position == 1 \
                            else mid_price - (half_spread * spread_factor)

            # Ende des Product-Loops
        return trade_logs

    @staticmethod
    def run_threshold_exit_strategy(
            df_backtest_substrate: pd.DataFrame,
            model_column: str,
            t_long: float,
            t_short: float,
            t_exit_long: float,
            t_exit_short: float,
            spread_factor: float = 0.5,
            trade_direction: Literal["both", "long_only", "short_only"] = 'both',
            **kwargs
    ) -> List[Dict]:

        trade_logs = []

        for product_id, group in df_backtest_substrate.groupby('delivery_start', observed=True):
            times = group['snapshot_times'].values
            signals = group[model_column].values

            # Preis-Vektoren
            sig_bids, sig_asks = group['signal_time_bid'].values, group['signal_time_ask'].values
            exc_bids, exc_asks = group['exec_bid'].values, group['exec_ask'].values

            current_position = 0
            entry_price, entry_time = 0.0, None
            entry_slippage, entry_spread_cost = 0.0, 0.0

            for i in range(len(group)):
                pred = signals[i]
                is_gate_closure = (i == len(group) - 1)

                # --- 1. EXIT LOGIK ---
                should_exit = False
                if current_position == 1 and (pred < t_exit_long or is_gate_closure):
                    should_exit = True
                elif current_position == -1 and (pred > t_exit_short or is_gate_closure):
                    should_exit = True

                if should_exit and current_position != 0:
                    mid_exc = (exc_asks[i] + exc_bids[i]) / 2
                    half_spread_exc = (exc_asks[i] - exc_bids[i]) / 2

                    # Realisierter Exit-Preis inkl. Kosten
                    exit_price = mid_exc - (half_spread_exc * spread_factor) if current_position == 1 \
                        else mid_exc + (half_spread_exc * spread_factor)

                    # Metriken berechnen
                    mid_sig = (sig_asks[i] + sig_bids[i]) / 2
                    exit_slippage = (mid_sig - mid_exc) if current_position == 1 else (mid_exc - mid_sig)
                    exit_spread_cost = half_spread_exc * spread_factor

                    pnl = (exit_price - entry_price) * current_position

                    trade_logs.append({
                        'delivery_start': product_id,
                        'entry_time': entry_time,
                        'exit_time': times[i],
                        'side': 'LONG' if current_position == 1 else 'SHORT',
                        'pnl': pnl,
                        'slippage_loss': entry_slippage + exit_slippage,
                        'spread_loss': entry_spread_cost + exit_spread_cost,  # Summe der Round-Trip Kosten
                        'reason': 'Gate Closure' if is_gate_closure else 'Threshold Exit'
                    })
                    current_position = 0

                # --- 2. ENTRY LOGIK ---
                if current_position == 0 and not is_gate_closure:
                    target_pos = 1 if pred > t_long else (-1 if pred < t_short else 0)
                    is_allowed = (target_pos == 1 and trade_direction in ['both', 'long_only']) or \
                                 (target_pos == -1 and trade_direction in ['both', 'short_only'])

                    if is_allowed:
                        current_position = target_pos
                        entry_time = times[i]

                        mid_exc = (exc_asks[i] + exc_bids[i]) / 2
                        half_spread_exc = (exc_asks[i] - exc_bids[i]) / 2

                        # Realisierter Einstiegspreis
                        entry_price = mid_exc + (half_spread_exc * spread_factor) if current_position == 1 \
                            else mid_exc - (half_spread_exc * spread_factor)

                        # Metriken isolieren
                        mid_sig = (sig_asks[i] + sig_bids[i]) / 2
                        entry_slippage = (mid_exc - mid_sig) if current_position == 1 else (mid_sig - mid_exc)
                        entry_spread_cost = half_spread_exc * spread_factor

        return trade_logs
