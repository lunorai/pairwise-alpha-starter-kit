import pandas as pd
import numpy as np

# --- ORIGINAL FUNCTIONS --- 
def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy: Buy LDO if BTC or ETH pumped >2% exactly 4 hours ago.

    Inputs:
    - candles_target: OHLCV for LDO (1H)
    - candles_anchor: Merged OHLCV with columns 'close_BTC' and 'close_ETH' (1H)

    Output:
    - DataFrame with ['timestamp', 'signal']
    """
    try:
        df = pd.merge(
            candles_target[['timestamp', 'close']],
            candles_anchor[['timestamp', 'close_BTC', 'close_ETH']],
            on='timestamp',
            how='inner'
        )

        if len(df) < 5:
            signals_df = pd.DataFrame(index=candles_target.index)
            signals_df['timestamp'] = candles_target['timestamp']
            signals_df['signal'] = 'HOLD'
            return signals_df[['timestamp', 'signal']]

        df['btc_return_4h_ago'] = df['close_BTC'].pct_change().shift(4)
        df['eth_return_4h_ago'] = df['close_ETH'].pct_change().shift(4)

        signals = []
        for i in range(len(df)):
            # Check for NaN from shift operation (especially at the beginning)
            btc_pump_val = df['btc_return_4h_ago'].iloc[i]
            eth_pump_val = df['eth_return_4h_ago'].iloc[i]

            btc_pump = pd.notna(btc_pump_val) and btc_pump_val > 0.02
            eth_pump = pd.notna(eth_pump_val) and eth_pump_val > 0.02
            
            if btc_pump or eth_pump:
                signals.append('BUY')
            else:
                signals.append('HOLD')

        df['signal'] = signals


        return df[['timestamp', 'signal']]

    except Exception as e:
        raise RuntimeError(f"Error in generate_signals: {e}")

def get_coin_metadata() -> dict:
    """
    Specifies the target and anchor coins used in this strategy.

    Returns:
    {
        "target": {"symbol": "LDO", "timeframe": "1H"},
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"}
        ]
    }
    """
    return {
        "target": {
            "symbol": "LDO",
            "timeframe": "1H"
        },
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"}
        ]
    }
# --- END OF ORIGINAL FUNCTIONS ---


# --- NEW FUNCTIONALITIES ---

def analyze_signals_custom(signals_df: pd.DataFrame) -> dict:
    """
    Analyzes the generated signals to count occurrences and frequencies.
    Input: DataFrame from generate_signals (columns: ['timestamp', 'signal']).
    Output: Dictionary with signal counts and frequencies.
    """
    if not isinstance(signals_df, pd.DataFrame) or 'signal' not in signals_df.columns:
        return {"error": "Invalid signals_df input"}
    
    signal_counts = signals_df['signal'].value_counts().to_dict()
    total_signals = len(signals_df)
    signal_frequencies = {k: v / total_signals for k, v in signal_counts.items()} if total_signals > 0 else {}
    
    return {
        "signal_counts": signal_counts,
        "signal_frequencies": signal_frequencies,
        "total_signals": total_signals
    }

def generate_strategy_report_custom(metadata: dict, signals_df: pd.DataFrame, analysis_results: dict = None) -> str:
    """
    Generates a simple string report summarizing strategy metadata and signal analysis.
    """
    report_lines = []
    report_lines.append("=== Strategy Report ===")
    
    # Metadata section
    report_lines.append("\n--- Metadata ---")
    if metadata and isinstance(metadata, dict):
        report_lines.append(f"Target Coin: {metadata.get('target', {}).get('symbol', 'N/A')}")
        report_lines.append(f"Target Timeframe: {metadata.get('target', {}).get('timeframe', 'N/A')}")
        anchors = metadata.get('anchors', [])
        if anchors:
            report_lines.append("Anchor Coins:")
            for anchor in anchors:
                report_lines.append(f"  - Symbol: {anchor.get('symbol', 'N/A')}, Timeframe: {anchor.get('timeframe', 'N/A')}")
        else:
            report_lines.append("Anchor Coins: None specified")
    else:
        report_lines.append("Metadata not available or in incorrect format.")

    # Signal Analysis Section
    report_lines.append("\n--- Signal Analysis ---")
    if analysis_results is None and isinstance(signals_df, pd.DataFrame):
        analysis_results = analyze_signals_custom(signals_df)
    
    if analysis_results and isinstance(analysis_results, dict) and 'error' not in analysis_results:
        report_lines.append(f"Total Signals Generated: {analysis_results.get('total_signals', 0)}")
        report_lines.append("Signal Counts:")
        for signal, count in analysis_results.get('signal_counts', {}).items():
            report_lines.append(f"  - {signal}: {count}")
        report_lines.append("Signal Frequencies:")
        for signal, freq in analysis_results.get('signal_frequencies', {}).items():
            report_lines.append(f"  - {signal}: {freq:.2%}")
    elif analysis_results and 'error' in analysis_results:
        report_lines.append(f"Could not analyze signals: {analysis_results['error']}")
    else:
        report_lines.append("Signal analysis data not available or signals_df not provided.")
        
    report_lines.append("\n=== End of Report ===")
    return "\n".join(report_lines)

def post_process_signals_custom(signals_df: pd.DataFrame, max_consecutive_buys: int = None, min_hold_after_buy: int = None) -> pd.DataFrame:
    """
    Applies post-processing rules to the original signals.
    - max_consecutive_buys: Limits the number of consecutive BUY signals.
    - min_hold_after_buy: Enforces a minimum number of HOLD periods after a BUY signal.
    """
    if not isinstance(signals_df, pd.DataFrame) or 'signal' not in signals_df.columns:
        # Or raise an error, or return signals_df unmodified
        print("Warning: Invalid input to post_process_signals_custom. Returning original signals.")
        return signals_df.copy()

    processed_signals = signals_df.copy()
    current_signal_col = processed_signals['signal'].values # Operate on numpy array for potential speed up

    # Rule 1: Max consecutive BUYs
    if max_consecutive_buys is not None and max_consecutive_buys > 0:
        consecutive_buys = 0
        for i in range(len(current_signal_col)):
            if current_signal_col[i] == 'BUY':
                consecutive_buys += 1
                if consecutive_buys > max_consecutive_buys:
                    current_signal_col[i] = 'HOLD' # Change to HOLD
            else:
                consecutive_buys = 0 # Reset counter
    
    # Rule 2: Min HOLD period after a BUY
    # This rule should be applied after the consecutive BUYs rule, or carefully considered in conjunction
    if min_hold_after_buy is not None and min_hold_after_buy > 0:
        last_buy_index = - (min_hold_after_buy + 1) # Initialize to allow immediate buy
        for i in range(len(current_signal_col)):
            if current_signal_col[i] == 'BUY':
                # If a BUY occurs too soon after a previous BUY (that wasn't changed to HOLD by rule 1)
                if i - last_buy_index <= min_hold_after_buy:
                     # This BUY is too soon, check if it was an original BUY or became BUY after rule 1
                     # This logic can get complex if BUYs can be re-instated. Simpler: if it's BUY now, and too soon, make HOLD.
                    current_signal_col[i] = 'HOLD'
                else:
                    last_buy_index = i # Record this valid BUY
            # If it's already HOLD or SELL, this rule doesn't apply to change it further based on *this* rule's logic.

    processed_signals['signal'] = current_signal_col
    return processed_signals


# --- MAIN EXECUTION BLOCK FOR TESTING ---
if __name__ == '__main__':
    print("===== Running Extended Local Tests for strategy.py =====")

    # --- Test Case 1: Original Example (from previous successful validation) ---
    print("\n--- Test Case 1: Basic Scenario ---")
    metadata_tc1 = get_coin_metadata() # Uses the original function
    print(f"Metadata: {metadata_tc1}")

    timestamps_tc1 = pd.to_datetime([
        '2025-01-01 00:00:00', '2025-01-01 01:00:00', '2025-01-01 02:00:00', 
        '2025-01-01 03:00:00', '2025-01-01 04:00:00', '2025-01-01 05:00:00',
        '2025-01-01 06:00:00', '2025-01-01 07:00:00'
    ], utc=True)
    
    candles_target_tc1_data = {
        'timestamp': timestamps_tc1,
        'open':  [1.0] * 8, 'high': [1.1] * 8, 'low': [0.9] * 8, 
        'close': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Target close doesn't affect this strategy
        'volume': [1000] * 8
    }
    ct_df_tc1 = pd.DataFrame(candles_target_tc1_data)

    # BTC pump at index 1 (0.03 return), should trigger BUY at index 1+4=5
    candles_anchor_tc1_data = {
        'timestamp': timestamps_tc1,
        'close_BTC': [100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'close_ETH': [10.0] * 8 # No ETH pump
    }
    ca_df_tc1 = pd.DataFrame(candles_anchor_tc1_data)

    print("\nOriginal Target Candles (TC1):")
    print(ct_df_tc1)
    print("\nOriginal Anchor Candles (TC1):")
    print(ca_df_tc1)

    original_signals_tc1 = generate_signals(ct_df_tc1, ca_df_tc1) # Uses the original function
    print("\nOriginal Generated Signals (TC1):")
    print(original_signals_tc1)

    # Test new auxiliary functions
    analysis_tc1 = analyze_signals_custom(original_signals_tc1)
    print("\nSignal Analysis (TC1):")
    print(analysis_tc1)

    report_tc1 = generate_strategy_report_custom(metadata_tc1, original_signals_tc1, analysis_tc1)
    print("\nStrategy Report (TC1):")
    print(report_tc1)

    # Test post-processing
    post_processed_signals_tc1 = post_process_signals_custom(original_signals_tc1, max_consecutive_buys=2, min_hold_after_buy=1)
    print("\nPost-Processed Signals (TC1 - max_consecutive_buys=2, min_hold_after_buy=1):")
    print(post_processed_signals_tc1)
    analysis_pp_tc1 = analyze_signals_custom(post_processed_signals_tc1)
    print("\nPost-Processed Signal Analysis (TC1):")
    print(analysis_pp_tc1)

    # --- Test Case 2: Insufficient data for lag ---
    print("\n--- Test Case 2: Insufficient Data for Lag ---")
    timestamps_tc2 = pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 01:00:00', '2025-01-01 02:00:00'], utc=True)
    ct_df_tc2 = pd.DataFrame({'timestamp': timestamps_tc2, 'close': [1.0]*3, 'open': [1.0]*3, 'high': [1.0]*3, 'low': [1.0]*3, 'volume': [1.0]*3})
    ca_df_tc2 = pd.DataFrame({'timestamp': timestamps_tc2, 'close_BTC': [100.0, 103.0, 100.0], 'close_ETH': [10.0]*3})
    
    original_signals_tc2 = generate_signals(ct_df_tc2, ca_df_tc2)
    print("\nOriginal Generated Signals (TC2 - Insufficient Data):")
    print(original_signals_tc2) # Expected all HOLDs due to internal check in generate_signals
    analysis_tc2 = analyze_signals_custom(original_signals_tc2)
    print(f"Analysis (TC2): {analysis_tc2}")

    # --- Test Case 3: Multiple BUY signals for post-processing test ---
    print("\n--- Test Case 3: Multiple BUYs for Post-Processing ---")
    timestamps_tc3 = pd.to_datetime([f'2025-01-01 {i:02d}:00:00' for i in range(10)], utc=True)
    ct_df_tc3 = pd.DataFrame({'timestamp': timestamps_tc3, 'close': [1.0]*10, 'open': [1.0]*10, 'high': [1.0]*10, 'low': [1.0]*10, 'volume': [1.0]*10})
    # BTC pumps at index 1 (signal at 5), 2 (signal at 6), 3 (signal at 7)
    # ETH pumps at index 4 (signal at 8)
    btc_closes_tc3 = [100.0] * 10
    btc_closes_tc3[1] = 103.0 # pump for signal at index 5
    btc_closes_tc3[2] = 106.0 # pump for signal at index 6 (relative to 103.0)
    btc_closes_tc3[3] = 109.0 # pump for signal at index 7 (relative to 106.0)
    eth_closes_tc3 = [10.0] * 10
    eth_closes_tc3[4] = 10.3 # pump for signal at index 8

    ca_df_tc3 = pd.DataFrame({'timestamp': timestamps_tc3, 'close_BTC': btc_closes_tc3, 'close_ETH': eth_closes_tc3})
    original_signals_tc3 = generate_signals(ct_df_tc3, ca_df_tc3)
    print("\nOriginal Generated Signals (TC3 - Multiple BUYs):")
    print(original_signals_tc3)
    analysis_orig_tc3 = analyze_signals_custom(original_signals_tc3)
    print(f"Analysis Original (TC3): {analysis_orig_tc3}")

    # Test max_consecutive_buys = 1
    pp_signals_tc3_rule1 = post_process_signals_custom(original_signals_tc3.copy(), max_consecutive_buys=1)
    print("\nPost-Processed Signals (TC3 - max_consecutive_buys=1):")
    print(pp_signals_tc3_rule1)
    analysis_pp_tc3_rule1 = analyze_signals_custom(pp_signals_tc3_rule1)
    print(f"Analysis Post-Processed (TC3 Rule 1): {analysis_pp_tc3_rule1}")

    # Test min_hold_after_buy = 2 (applied to original signals for this test)
    pp_signals_tc3_rule2 = post_process_signals_custom(original_signals_tc3.copy(), min_hold_after_buy=2)
    print("\nPost-Processed Signals (TC3 - min_hold_after_buy=2):")
    print(pp_signals_tc3_rule2)
    analysis_pp_tc3_rule2 = analyze_signals_custom(pp_signals_tc3_rule2)
    print(f"Analysis Post-Processed (TC3 Rule 2): {analysis_pp_tc3_rule2}")
    
    # Test combined rules
    pp_signals_tc3_combined = post_process_signals_custom(original_signals_tc3.copy(), max_consecutive_buys=2, min_hold_after_buy=1)
    print("\nPost-Processed Signals (TC3 - max_consecutive_buys=2, min_hold_after_buy=1):")
    print(pp_signals_tc3_combined)
    analysis_pp_tc3_combined = analyze_signals_custom(pp_signals_tc3_combined)
    print(f"Analysis Post-Processed (TC3 Combined): {analysis_pp_tc3_combined}")