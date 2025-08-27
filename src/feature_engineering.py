"""
Feature engineering for stock price prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

class FeatureEngineer:
    """Feature engineering for stock price prediction"""
    
    def __init__(self):
        self.technical_indicators = []
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        df = df.copy()
        
        # Price changes
        df['price_change'] = df['Close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # High-Low spread
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['hl_spread_pct'] = df['hl_spread'] * 100
        
        # Open-Close spread
        df['oc_spread'] = (df['Close'] - df['Open']) / df['Open']
        
        # Price ranges
        df['price_range'] = df['High'] - df['Low']
        df['price_range_pct'] = df['price_range'] / df['Close'] * 100
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame, 
                           windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Add moving averages"""
        df = df.copy()
        
        for window in windows:
            df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ma_{window}_pct'] = (df['Close'] - df[f'ma_{window}']) / df[f'ma_{window}']
        
        return df
    
    def add_exponential_moving_averages(self, df: pd.DataFrame, 
                                      periods: List[int] = [12, 26]) -> pd.DataFrame:
        """Add exponential moving averages"""
        df = df.copy()
        
        for period in periods:
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'ema_{period}_pct'] = (df['Close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, 
                           window: int = 20, 
                           num_std: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=window).mean()
        df['bb_std'] = df['Close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * num_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * num_std)
        
        # Bollinger Band features
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = df['bb_width'].rolling(window=20).mean()
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        df = df.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI features
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
        
        return df
    
    def add_macd(self, df: pd.DataFrame, 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)"""
        df = df.copy()
        
        # Calculate MACD
        ema_fast = df['Close'].ewm(span=fast_period).mean()
        ema_slow = df['Close'].ewm(span=slow_period).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD features
        df['macd_cross_above'] = ((df['macd'] > df['macd_signal']) & 
                                 (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_below'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def add_stochastic(self, df: pd.DataFrame, 
                      k_period: int = 14, 
                      d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        df = df.copy()
        
        # Calculate Stochastic
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic features
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        return df
    
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        df = df.copy()
        
        # Calculate True Range
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        df['atr_pct'] = df['atr'] / df['Close'] * 100
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        df = df.copy()
        
        # Rate of Change
        df['roc'] = df['Close'].pct_change(periods=10) * 100
        
        # Williams %R
        highest_high = df['High'].rolling(window=14).max()
        lowest_low = df['Low'].rolling(window=14).min()
        df['williams_r'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        
        # Commodity Channel Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        df = df.copy()
        
        # Rolling volatility
        df['volatility_5d'] = df['Close'].pct_change().rolling(window=5).std() * np.sqrt(252)
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['volatility_60d'] = df['Close'].pct_change().rolling(window=60).std() * np.sqrt(252)
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
        
        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['High'] / df['Low']) ** 2).rolling(window=20).mean()
        ))
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Extract date components
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Market session features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 28).astype(int)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                        columns: List[str] = None, 
                        lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features"""
        df = df.copy()
        
        if columns is None:
            columns = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, 
                           columns: List[str] = None,
                           windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling window features"""
        df = df.copy()
        
        if columns is None:
            columns = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all available features"""
        print("Creating price features...")
        df = self.add_price_features(df)
        
        print("Creating moving averages...")
        df = self.add_moving_averages(df)
        df = self.add_exponential_moving_averages(df)
        
        print("Creating Bollinger Bands...")
        df = self.add_bollinger_bands(df)
        
        print("Creating RSI...")
        df = self.add_rsi(df)
        
        print("Creating MACD...")
        df = self.add_macd(df)
        
        print("Creating Stochastic...")
        df = self.add_stochastic(df)
        
        print("Creating ATR...")
        df = self.add_atr(df)
        
        print("Creating momentum indicators...")
        df = self.add_momentum_indicators(df)
        
        print("Creating volatility features...")
        df = self.add_volatility_features(df)
        
        print("Creating time features...")
        df = self.add_time_features(df)
        
        print("Creating lag features...")
        df = self.add_lag_features(df)
        
        print("Creating rolling features...")
        df = self.add_rolling_features(df)
        
        # Remove any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, 
                                     target_column: str = 'Close_future') -> pd.DataFrame:
        """Get feature importance ranking using correlation"""
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found in dataframe")
            return pd.DataFrame()
        
        # Calculate correlations with target
        correlations = df.corr()[target_column].abs().sort_values(ascending=False)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        return feature_importance
    
    def select_top_features(self, df: pd.DataFrame, 
                          target_column: str = 'Close_future',
                          top_n: int = 50) -> pd.DataFrame:
        """Select top N features based on correlation with target"""
        feature_importance = self.get_feature_importance_ranking(df, target_column)
        
        # Get top N features (excluding target column)
        top_features = feature_importance[feature_importance['feature'] != target_column].head(top_n)
        
        # Select only top features and target
        selected_columns = list(top_features['feature']) + [target_column]
        selected_df = df[selected_columns].copy()
        
        print(f"Selected top {len(top_features)} features")
        return selected_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create all features
    featured_data = fe.create_all_features(sample_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"With features: {featured_data.shape}")
    print(f"Number of features created: {featured_data.shape[1] - sample_data.shape[1]}")
