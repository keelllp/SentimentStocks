"""
Production predictor — loads trained LightGBM artifacts and serves predictions.
Falls back to moving-average drift if models are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')
sys.path.insert(0, _BACKEND_DIR)

from feature_pipeline import FEATURE_COLUMNS, engineer_features_single, select_features

_log = logging.getLogger('sentimentstocks')

# Sentiment keys expected in sentiment_dict
_SENTIMENT_KEYS = ('Positive_mean', 'Negative_mean', 'Neutral_mean', 'news_count',
                   'news_count_7d_avg', 'news_count_30d_avg')

_NEUTRAL_SENTIMENT = {
    'Positive_mean': 0.0, 'Negative_mean': 0.0, 'Neutral_mean': 1.0,
    'news_count': 1, 'news_count_7d_avg': 1.0, 'news_count_30d_avg': 1.0,
}


def _max_price_cap() -> float:
    return 0.15  # ±15% max single prediction


class ProductionPredictor:
    """
    Loads direction_model.pkl + magnitude_model.pkl (new LightGBM stack).
    Automatically falls back to old xgb_model.pkl / rf_model.pkl if not found.
    """

    def __init__(self):
        self.direction_model = None
        self.magnitude_model = None
        self.label_encoder = None
        self.metadata: dict = {}
        self.model_name: str = 'LightGBM (not loaded)'
        self.horizon: int = 3
        self._use_legacy: bool = False
        # Legacy fallback
        self._xgb_model = None
        self._rf_model = None
        self._scaler = None
        self._legacy_features = None
        self._load_models()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        dir_path = os.path.join(_MODELS_DIR, 'direction_model.pkl')
        mag_path = os.path.join(_MODELS_DIR, 'magnitude_model.pkl')
        le_path = os.path.join(_MODELS_DIR, 'label_encoder.pkl')
        meta_path = os.path.join(_MODELS_DIR, 'model_metadata.json')

        if os.path.exists(dir_path) and os.path.exists(mag_path):
            try:
                self.direction_model = joblib.load(dir_path)
                self.magnitude_model = joblib.load(mag_path)
                _log.info('Loaded LightGBM direction + magnitude models')
            except Exception as e:
                _log.error(f'Failed to load LightGBM models: {e}')
                self.direction_model = None
                self.magnitude_model = None

        if os.path.exists(le_path):
            try:
                self.label_encoder = joblib.load(le_path)
            except Exception:
                pass

        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.model_name = self.metadata.get('model_name', 'LightGBM Ensemble')
                self.horizon = self.metadata.get('horizon_days', 3)
            except Exception:
                pass

        # If new models loaded successfully, we're done
        if self.direction_model is not None:
            return

        # Legacy fallback: try old XGB+RF pkl pair
        _log.warning('New LightGBM models not found — attempting legacy XGB/RF fallback')
        self._try_load_legacy()

    def _try_load_legacy(self) -> None:
        xgb_path = os.path.join(_MODELS_DIR, 'xgb_model.pkl')
        rf_path = os.path.join(_MODELS_DIR, 'rf_model.pkl')
        sc_path = os.path.join(_MODELS_DIR, 'scaler.pkl')
        le_path = os.path.join(_MODELS_DIR, 'label_encoder.pkl')

        if not (os.path.exists(xgb_path) and os.path.exists(rf_path)):
            _log.warning('No model artifacts found — will use moving-average fallback')
            return

        try:
            self._xgb_model = joblib.load(xgb_path)
            self._rf_model = joblib.load(rf_path)
            if os.path.exists(sc_path):
                self._scaler = joblib.load(sc_path)
            if os.path.exists(le_path) and self.label_encoder is None:
                self.label_encoder = joblib.load(le_path)
            self._legacy_features = getattr(self._xgb_model, 'feature_names_in_', None)
            self._use_legacy = True
            self.model_name = 'XGBoost + Random Forest (legacy)'
            _log.info('Loaded legacy XGBoost + Random Forest models')
        except Exception as e:
            _log.error(f'Legacy model load failed: {e}')

    # ── Feature preparation ────────────────────────────────────────────────────

    def _encode_stock(self, stock_name: str) -> int:
        if self.label_encoder is None:
            return 0
        try:
            return int(self.label_encoder.transform([stock_name.upper()])[0])
        except ValueError:
            _log.warning(f'Stock {stock_name} unknown to label encoder, using 0')
            return 0

    def _prepare_features_new(
        self, stock_name: str, stock_data: pd.DataFrame, sentiment_dict: dict
    ) -> Optional[pd.DataFrame]:
        """Prepare features for the LightGBM model stack."""
        try:
            df = stock_data.copy()

            # Attach sentiment columns
            for k in _SENTIMENT_KEYS:
                df[k] = float(sentiment_dict.get(k, _NEUTRAL_SENTIMENT.get(k, 0.0)))

            enc = self._encode_stock(stock_name)
            df = engineer_features_single(df, stock_encoded=enc)

            latest = df.iloc[[-1]].copy()
            return select_features(latest)
        except Exception as e:
            _log.error(f'Feature prep failed for {stock_name}: {e}')
            return None

    def _prepare_features_legacy(
        self, stock_name: str, stock_data: pd.DataFrame, sentiment_dict: dict
    ) -> Optional[pd.DataFrame]:
        """Prepare features for the old XGB/RF model (original schema)."""
        try:
            df = stock_data.copy()
            sentiment_score = sentiment_dict.get('Positive_mean', 0.0)

            if 'Volume' not in df.columns:
                df['Volume'] = 1_000_000

            df['Positive_mean'] = sentiment_score if sentiment_score > 0 else 0.0
            df['Negative_mean'] = abs(sentiment_score) if sentiment_score < 0 else 0.0
            df['Neutral_mean'] = 1.0 - (df['Positive_mean'] + df['Negative_mean'])
            df['news_count'] = 1
            df['news_count_7d_avg'] = 1
            df['news_count_30d_avg'] = 1

            for lag in [1, 2, 3]:
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
                df[f'Pos_sentiment_lag_{lag}'] = df['Positive_mean'].shift(lag)
                df[f'news_count_lag_{lag}'] = df['news_count'].shift(lag)

            df['Daily_Return'] = df['Close'].pct_change()
            df['SMA_7'] = df['Close'].rolling(7).mean()
            df['SMA_14'] = df['Close'].rolling(14).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Volatility_7d'] = df['Daily_Return'].rolling(7).std()
            df = df.bfill().fillna(0)

            if self.label_encoder is not None:
                try:
                    df['Stock_Encoded'] = int(self.label_encoder.transform([stock_name.upper()])[0])
                except ValueError:
                    df['Stock_Encoded'] = 0
            else:
                df['Stock_Encoded'] = 0

            latest = df.iloc[[-1]].copy()

            if self._legacy_features is not None:
                for f in self._legacy_features:
                    if f not in latest.columns:
                        latest[f] = 0.0
                return latest[self._legacy_features]
            return latest
        except Exception as e:
            _log.error(f'Legacy feature prep failed: {e}')
            return None

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        stock_name: str,
        stock_data: pd.DataFrame,
        sentiment_dict: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Predict 3-day direction + price for `stock_name`.

        Parameters
        ----------
        stock_name    : e.g. 'INFY'
        stock_data    : DataFrame with at least Close, Open, High, Low, Volume columns,
                        sorted chronologically (index can be DatetimeIndex or integer)
        sentiment_dict: optional dict from sentiment_live.get_live_sentiment().
                        Falls back to neutral if None.

        Returns dict with current_price, predicted_price, price_change, price_change_pct,
        confidence (float 0-1), signal, features_used, method, horizon.
        """
        if sentiment_dict is None:
            sentiment_dict = dict(_NEUTRAL_SENTIMENT)

        try:
            current_price = float(stock_data['Close'].iloc[-1])
        except Exception:
            _log.error('stock_data has no Close column or is empty')
            return None

        # ── LightGBM path ──────────────────────────────────────────────────────
        if self.direction_model is not None and self.magnitude_model is not None:
            features = self._prepare_features_new(stock_name, stock_data, sentiment_dict)
            if features is not None:
                try:
                    p_up = float(self.direction_model.predict_proba(features)[0, 1])
                    raw_magnitude = float(self.magnitude_model.predict(features)[0])

                    signal = 'BUY' if p_up >= 0.5 else 'SELL'
                    direction = 1 if signal == 'BUY' else -1
                    pct_change = direction * min(abs(raw_magnitude), _max_price_cap())

                    predicted_price = round(current_price * (1 + pct_change), 2)
                    return {
                        'current_price': round(current_price, 2),
                        'predicted_price': predicted_price,
                        'price_change': round(predicted_price - current_price, 2),
                        'price_change_pct': round(pct_change * 100, 2),
                        'confidence': round(p_up, 4),
                        'signal': signal,
                        'features_used': len(FEATURE_COLUMNS),
                        'method': 'LightGBM Direction + Magnitude',
                        'horizon': f'{self.horizon} Days',
                    }
                except Exception as e:
                    _log.error(f'LightGBM prediction failed: {e}')

        # ── Legacy XGB+RF path ─────────────────────────────────────────────────
        if self._use_legacy and self._xgb_model is not None and self._rf_model is not None:
            features = self._prepare_features_legacy(stock_name, stock_data, sentiment_dict)
            if features is not None:
                try:
                    if self._scaler is not None:
                        features_scaled = pd.DataFrame(
                            self._scaler.transform(features), columns=features.columns
                        )
                    else:
                        features_scaled = features

                    xgb_ret = float(self._xgb_model.predict(features_scaled)[0])
                    rf_ret = float(self._rf_model.predict(features_scaled)[0])
                    ensemble_ret = (xgb_ret + rf_ret) / 2.0

                    # Use volatility-based magnitude (original approach)
                    try:
                        vol = float(features_scaled.get('Volatility_7d', pd.Series([0.015])).iloc[-1])
                        if np.isnan(vol) or vol <= 0:
                            vol = 0.015
                    except Exception:
                        vol = 0.015

                    signal = 'BUY' if ensemble_ret > 0 else 'SELL'
                    pct_change = (vol * np.sqrt(3)) if signal == 'BUY' else -(vol * np.sqrt(3))
                    pct_change = max(min(pct_change, 0.12), -0.12)
                    predicted_price = round(current_price * (1 + pct_change), 2)

                    return {
                        'current_price': round(current_price, 2),
                        'predicted_price': predicted_price,
                        'price_change': round(predicted_price - current_price, 2),
                        'price_change_pct': round(pct_change * 100, 2),
                        'confidence': 0.55,
                        'signal': signal,
                        'features_used': int(features.shape[1]),
                        'method': 'XGBoost + RF (legacy)',
                        'horizon': '3 Days',
                    }
                except Exception as e:
                    _log.error(f'Legacy prediction failed: {e}')

        # ── Moving-average fallback ────────────────────────────────────────────
        _log.warning(f'Using moving-average fallback for {stock_name}')
        try:
            ma5 = stock_data['Close'].rolling(5, min_periods=1).mean().iloc[-1]
            ma20 = stock_data['Close'].rolling(20, min_periods=1).mean().iloc[-1]
            trend = float((ma5 - ma20) / ma20) if ma20 != 0 else 0.0
            trend = max(min(trend, 0.05), -0.05)
            predicted_price = round(current_price * (1 + trend), 2)
            return {
                'current_price': round(current_price, 2),
                'predicted_price': predicted_price,
                'price_change': round(predicted_price - current_price, 2),
                'price_change_pct': round(trend * 100, 2),
                'confidence': 0.50,
                'signal': 'BUY' if trend > 0 else 'SELL',
                'features_used': 2,
                'method': 'Moving Average (fallback)',
                'horizon': '3 Days',
            }
        except Exception as e:
            _log.error(f'Fallback prediction failed: {e}')
            return None
