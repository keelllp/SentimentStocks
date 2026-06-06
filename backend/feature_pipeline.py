"""
Shared feature engineering pipeline — imported by both train_model.py and
production_predictor.py so training and serving can never diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON = 3  # trading days ahead


# ──────────────────────────────────────────────
# Helper indicators
# ──────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


# ──────────────────────────────────────────────
# Core engineering (single-stock)
# ──────────────────────────────────────────────

def engineer_features_single(
    df: pd.DataFrame,
    stock_encoded: int = 0,
) -> pd.DataFrame:
    """
    Compute all model features for ONE stock's OHLCV + sentiment DataFrame.

    Required input columns (all numeric, NaN-free before call):
        Open, High, Low, Close, Volume,
        Positive_mean, Neutral_mean, Negative_mean,
        news_count, news_count_7d_avg, news_count_30d_avg

    Returns a new DataFrame (same index) with FEATURE_COLUMNS columns added.
    Rows with NaN from rolling/shift are back-filled then zero-filled.
    """
    d = df.copy()

    # Ensure all source columns exist and are numeric
    _src = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Positive_mean', 'Neutral_mean', 'Negative_mean',
        'news_count', 'news_count_7d_avg', 'news_count_30d_avg',
    ]
    for col in _src:
        if col not in d.columns:
            d[col] = 0.0
        else:
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0.0)

    # ── Lags ──────────────────────────────────
    for lag in [1, 2, 3]:
        d[f'Close_lag_{lag}'] = d['Close'].shift(lag)
        d[f'Volume_lag_{lag}'] = d['Volume'].shift(lag)
        d[f'Pos_sentiment_lag_{lag}'] = d['Positive_mean'].shift(lag)
        d[f'news_count_lag_{lag}'] = d['news_count'].shift(lag)

    # ── Classic technical ─────────────────────
    d['Daily_Return'] = d['Close'].pct_change()
    d['SMA_7'] = d['Close'].rolling(7, min_periods=1).mean()
    d['SMA_14'] = d['Close'].rolling(14, min_periods=1).mean()
    d['EMA_12'] = d['Close'].ewm(span=12, adjust=False).mean()
    d['EMA_26'] = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = d['EMA_12'] - d['EMA_26']
    d['Volatility_7d'] = d['Daily_Return'].rolling(7, min_periods=2).std()

    # ── New technical indicators ──────────────
    d['RSI_14'] = _rsi(d['Close'], 14)
    d['ATR_14'] = _atr(d['High'], d['Low'], d['Close'], 14)

    sma20 = d['Close'].rolling(20, min_periods=5).mean()
    std20 = d['Close'].rolling(20, min_periods=5).std().replace(0, np.nan)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    d['BB_pct'] = (d['Close'] - lower) / band_width  # 0=at lower, 1=at upper

    d['ROC_5'] = d['Close'].pct_change(5)
    d['ROC_10'] = d['Close'].pct_change(10)

    d['Close_SMA7_ratio'] = d['Close'] / d['SMA_7'].replace(0, np.nan)
    d['Close_SMA14_ratio'] = d['Close'] / d['SMA_14'].replace(0, np.nan)

    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist'] = d['MACD'] - d['MACD_signal']

    vol_mean20 = d['Volume'].rolling(20, min_periods=5).mean()
    vol_std20 = d['Volume'].rolling(20, min_periods=5).std().replace(0, np.nan)
    d['Volume_zscore_20'] = (d['Volume'] - vol_mean20) / vol_std20

    # ── Longer-term trend features ────────────
    d['SMA_50'] = d['Close'].rolling(50, min_periods=20).mean()
    d['SMA_200'] = d['Close'].rolling(200, min_periods=80).mean()
    d['Close_SMA50_ratio'] = d['Close'] / d['SMA_50'].replace(0, np.nan)
    d['Close_SMA200_ratio'] = d['Close'] / d['SMA_200'].replace(0, np.nan)

    high_252 = d['High'].rolling(252, min_periods=60).max()
    low_252 = d['Low'].rolling(252, min_periods=60).min()
    range_252 = (high_252 - low_252).replace(0, np.nan)
    d['Price_pct_252d'] = (d['Close'] - low_252) / range_252  # 0=52w-low, 1=52w-high

    # Intraday range as % of close (daily volatility proxy)
    d['High_Low_pct'] = (d['High'] - d['Low']) / d['Close'].replace(0, np.nan)

    # ── Sentiment momentum ────────────────────
    d['Sentiment_momentum_7'] = (
        d['Positive_mean'].rolling(7, min_periods=1).mean().diff()
    )
    # Lagged negative sentiment (bad news can persist differently to good news)
    for lag in [1, 2, 3]:
        d[f'Neg_sentiment_lag_{lag}'] = d['Negative_mean'].shift(lag)

    # ── Calendar ──────────────────────────────
    if isinstance(d.index, pd.DatetimeIndex):
        d['DayOfWeek'] = d.index.dayofweek.astype(float)
    elif 'Date' in d.columns:
        d['DayOfWeek'] = pd.to_datetime(d['Date']).dt.dayofweek.astype(float)
    else:
        d['DayOfWeek'] = 2.0  # Wednesday as neutral default

    # ── Stock identity ────────────────────────
    d['Stock_Encoded'] = float(stock_encoded)

    # Fill NaN from rolling/shift (bfill first to avoid leading-edge issues)
    d = d.bfill().fillna(0.0)

    return d


# ──────────────────────────────────────────────
# Canonical ordered feature list
# ──────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Raw OHLCV
    'Open', 'High', 'Low', 'Close', 'Volume',
    # Sentiment base
    'news_count', 'Positive_mean', 'Neutral_mean', 'Negative_mean',
    'news_count_7d_avg', 'news_count_30d_avg',
    # Lags (price, volume, sentiment, news)
    'Close_lag_1', 'Volume_lag_1', 'Pos_sentiment_lag_1', 'news_count_lag_1',
    'Close_lag_2', 'Volume_lag_2', 'Pos_sentiment_lag_2', 'news_count_lag_2',
    'Close_lag_3', 'Volume_lag_3', 'Pos_sentiment_lag_3', 'news_count_lag_3',
    # Classic technical
    'Daily_Return', 'SMA_7', 'SMA_14', 'EMA_12', 'EMA_26', 'MACD', 'Volatility_7d',
    # New technical
    'RSI_14', 'ATR_14', 'BB_pct',
    'ROC_5', 'ROC_10',
    'Close_SMA7_ratio', 'Close_SMA14_ratio',
    'MACD_signal', 'MACD_hist',
    'Volume_zscore_20',
    # Sentiment derived
    'Sentiment_momentum_7',
    # Lagged negative sentiment
    'Neg_sentiment_lag_1', 'Neg_sentiment_lag_2', 'Neg_sentiment_lag_3',
    # Longer-term trend
    'SMA_50', 'SMA_200', 'Close_SMA50_ratio', 'Close_SMA200_ratio', 'Price_pct_252d',
    # Intraday range
    'High_Low_pct',
    # Calendar
    'DayOfWeek',
    # Identity
    'Stock_Encoded',
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select FEATURE_COLUMNS in canonical order, filling any missing columns with 0."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS].copy()


# ──────────────────────────────────────────────
# Multi-stock training dataset builder
# ──────────────────────────────────────────────

def build_training_dataset(
    df_all: pd.DataFrame,
    label_encoder,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build per-row features + two targets from the full multi-stock DataFrame.

    df_all must have: Stock, Datetime, Open, High, Low, Close, Volume,
                      Positive_mean, Neutral_mean, Negative_mean,
                      news_count, news_count_7d_avg, news_count_30d_avg

    Returns (X, y_direction, y_return, datetimes):
        X          : DataFrame with FEATURE_COLUMNS, sorted chronologically
        y_direction: binary Series (1=up, 0=down) HORIZON days ahead
        y_return   : continuous fractional return HORIZON days ahead
        datetimes  : aligned Datetime series (for time-split)
    """
    stock_enc_map = {s: int(label_encoder.transform([s])[0]) for s in df_all['Stock'].unique()}

    records = []
    for stock, grp in df_all.groupby('Stock'):
        grp = grp.sort_values('Datetime').reset_index(drop=True)

        # Build targets BEFORE engineering features (no data from the future in features)
        fwd_close = grp['Close'].shift(-HORIZON)
        grp['_direction'] = (fwd_close > grp['Close']).astype(float)
        grp['_return'] = (fwd_close - grp['Close']) / grp['Close'].replace(0, np.nan)

        grp = grp.dropna(subset=['_direction', '_return'])
        if len(grp) < 30:
            continue

        enc = stock_enc_map.get(stock, 0)
        grp = engineer_features_single(grp, stock_encoded=enc)
        records.append(grp)

    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values('Datetime').reset_index(drop=True)

    X = select_features(combined)
    y_dir = combined['_direction'].astype(int)
    y_ret = combined['_return'].astype(float)
    dts = combined['Datetime']

    return X, y_dir, y_ret, dts
