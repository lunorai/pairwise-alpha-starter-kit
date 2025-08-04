#!/usr/bin/env python3
"""
Cryptocurrency Data Manager
===========================

A simple data manager to fetch OHLCV data from Binance for specified symbols and timeframes,
then normalize everything to 1H frequency for analysis.

Input format: [{"symbol": "BTC", "timeframe": "1H"}, {"symbol": "ETH", "timeframe": "2H"}]
Output: Single DataFrame with 1H frequency and columns like close_BTC_1H, volume_ETH_2H, etc.
"""

import pandas as pd
import requests
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class CryptoDataManager:
    """
    Simple data manager for fetching and normalizing cryptocurrency data.
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limit_delay = 0.1
        self.max_retries = 3
        
        # Round 4 date range
        self.start_date = "2025-01-01 00:00:00"
        self.end_date = "2025-06-29 23:59:59"
        
        # Binance interval mapping
        self.interval_map = {
            "1H": "1h",
            "2H": "2h", 
            "4H": "4h",
            "12H": "12h",
            "1D": "1d"
        }
    
    def fetch_binance_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance for a specific symbol and timeframe.
        
        Args:
            symbol: Base symbol (e.g., 'BTC')
            timeframe: Timeframe ('1H', '2H', '4H', '12H', '1D')
            
        Returns:
            DataFrame with datetime, open, high, low, close, volume columns
        """
        usdt_pair = f"{symbol}USDT"
        binance_interval = self.interval_map.get(timeframe)
        
        if not binance_interval:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        logging.info(f"Fetching {usdt_pair} data for {timeframe} timeframe...")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S")
                      .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(self.end_date, "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_data = []
        current_start = start_ts
        
        # Calculate chunk size based on timeframe to stay under 1000 limit
        timeframe_hours = {"1H": 1, "2H": 2, "4H": 4, "12H": 12, "1D": 24}
        hours_per_chunk = timeframe_hours[timeframe] * 999  # Leave some buffer
        chunk_size_ms = hours_per_chunk * 60 * 60 * 1000
        
        chunk_num = 0
        while current_start < end_ts:
            chunk_num += 1
            current_end = min(current_start + chunk_size_ms, end_ts)
            
            logging.info(f"  Downloading chunk {chunk_num} for {usdt_pair}...")
            
            chunk_data = self._download_chunk(usdt_pair, binance_interval, current_start, current_end)
            
            if chunk_data:
                for kline in chunk_data:
                    try:
                        timestamp = int(kline[0])
                        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                        
                        all_data.append({
                            'timestamp': dt,
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        })
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping invalid kline: {e}")
            
            current_start = current_end + 1
            time.sleep(self.rate_limit_delay)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            logging.info(f"Downloaded {len(df)} data points for {symbol} {timeframe}")
        else:
            logging.warning(f"No data retrieved for {symbol} {timeframe}")
        
        return df
    
    def _download_chunk(self, symbol: str, interval: str, start_time: int, end_time: int) -> Optional[list]:
        """Download a single chunk with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/klines",
                    params={
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': start_time,
                        'endTime': end_time,
                        'limit': 1000
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    logging.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"API error {response.status_code}: {response.text}")
                    
            except requests.RequestException as e:
                logging.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def normalize_to_1h_frequency(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Normalize all downloaded data to 1H frequency and combine into single DataFrame.
        
        Args:
            data_dict: Dictionary in format {symbol: {timeframe: dataframe}}
            
        Returns:
            Single DataFrame with 1H frequency and properly named columns
        """
        logging.info("Normalizing data to 1H frequency...")
        
        # Create 1H frequency timestamp index for the entire date range
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        
        # Create hourly range
        hourly_index = pd.date_range(start=start_dt, end=end_dt, freq='1H', tz=timezone.utc)
        
        # Initialize result DataFrame with timestamp index
        result_df = pd.DataFrame(index=hourly_index)
        result_df.index.name = 'timestamp'
        
        # Process each symbol and timeframe
        for symbol, timeframe_data in data_dict.items():
            for timeframe, df in timeframe_data.items():
                if df.empty:
                    logging.warning(f"No data for {symbol} {timeframe}, filling with NaN")
                    continue
                
                # Set timestamp as index for resampling
                df_indexed = df.set_index('timestamp')
                
                # Rename columns with the required format
                column_mapping = {
                    'open': f'open_{symbol}_{timeframe}',
                    'high': f'high_{symbol}_{timeframe}',
                    'low': f'low_{symbol}_{timeframe}',
                    'close': f'close_{symbol}_{timeframe}',
                    'volume': f'volume_{symbol}_{timeframe}'
                }
                df_renamed = df_indexed.rename(columns=column_mapping)
                
                # Reindex to 1H frequency (this automatically fills missing hours with NaN)
                df_reindexed = df_renamed.reindex(hourly_index)
                
                # Add to result DataFrame
                for col in df_reindexed.columns:
                    result_df[col] = df_reindexed[col]
                
                logging.info(f"Added {symbol} {timeframe} data with {len(df_renamed)} non-NaN rows")
        
        # Reset index to make timestamp a column
        result_df = result_df.reset_index()
        
        logging.info(f"Final DataFrame shape: {result_df.shape}")
        return result_df
    
    def get_market_data(self, symbol_configs: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Main runner function to fetch and normalize market data.
        
        Args:
            symbol_configs: List of dicts like [{"symbol": "BTC", "timeframe": "1H"}]
            
        Returns:
            Single DataFrame with 1H frequency and all symbols combined
        """
        logging.info(f"Starting data download for {len(symbol_configs)} symbol/timeframe combinations")
        
        # Validate inputs
        valid_timeframes = set(self.interval_map.keys())
        for config in symbol_configs:
            if 'symbol' not in config or 'timeframe' not in config:
                raise ValueError(f"Invalid config format: {config}")
            if config['timeframe'] not in valid_timeframes:
                raise ValueError(f"Invalid timeframe {config['timeframe']}. Must be one of {valid_timeframes}")
        
        # Download data for each symbol/timeframe combination
        data_dict = {}
        
        for config in symbol_configs:
            symbol = config['symbol']
            timeframe = config['timeframe']
            
            if symbol not in data_dict:
                data_dict[symbol] = {}
            
            # Fetch data from Binance
            df = self.fetch_binance_data(symbol, timeframe)
            data_dict[symbol][timeframe] = df
        
        # Normalize and combine all data
        final_df = self.normalize_to_1h_frequency(data_dict)
        
        logging.info("Data processing completed successfully")
        return final_df
