"""
Production Stock Price Predictor
Uses trained LightGBM model for accurate predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProductionPredictor:
    """Production predictor using trained LightGBM model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_info = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and scaler"""
        try:
            # Load the trained LightGBM model
            model_path = '../models/optimized_lgb_model.joblib'
            print(f"🔍 Looking for model at: {model_path}")
            print(f"🔍 Current working directory: {os.getcwd()}")
            print(f"🔍 Model exists: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✅ Loaded trained LightGBM model")
            else:
                print("❌ Trained model not found")
                return False
            
            # Load the scaler
            scaler_path = '../models/enhanced_robust_scaler.joblib'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded data scaler")
            else:
                print("❌ Scaler not found")
                return False
            
            # Load feature info
            feature_path = '../models/feature_info.joblib'
            if os.path.exists(feature_path):
                self.feature_info = joblib.load(feature_path)
                print("✅ Loaded feature information")
            else:
                print("❌ Feature info not found")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def prepare_features(self, stock_data, sentiment_score=0.0):
        """Prepare features for prediction"""
        try:
            if stock_data is None or stock_data.empty:
                return None
            
            # Calculate technical indicators
            df = stock_data.copy()
            
            # Price features
            df['price_change'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['volume_price_ratio'] = df['Volume'] / df['Close']
            
            # Moving averages
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_10'] = df['Close'].rolling(10).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['ma_50'] = df['Close'].rolling(50).mean()
            
            # Price relative to moving averages
            df['price_ma5_ratio'] = df['Close'] / df['ma_5']
            df['price_ma10_ratio'] = df['Close'] / df['ma_10']
            df['price_ma20_ratio'] = df['Close'] / df['ma_20']
            df['price_ma50_ratio'] = df['Close'] / df['ma_50']
            
            # Volatility
            df['volatility_5'] = df['Close'].rolling(5).std()
            df['volatility_10'] = df['Close'].rolling(10).std()
            df['volatility_20'] = df['Close'].rolling(20).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Volume features
            df['volume_ma_5'] = df['Volume'].rolling(5).mean()
            df['volume_ma_10'] = df['Volume'].rolling(10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_10']
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            
            # Get the latest row (most recent data)
            latest = df.iloc[-1:].copy()
            
            # Add sentiment score
            latest['sentiment_score'] = sentiment_score
            
            # Select features (remove non-feature columns)
            feature_columns = [
                'price_change', 'high_low_ratio', 'volume_price_ratio',
                'ma_5', 'ma_10', 'ma_20', 'ma_50',
                'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio', 'price_ma50_ratio',
                'volatility_5', 'volatility_10', 'volatility_20',
                'rsi', 'bb_width', 'bb_position',
                'macd', 'macd_signal', 'macd_histogram',
                'volume_ma_5', 'volume_ma_10', 'volume_ratio',
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
                'sentiment_score'
            ]
            
            # Filter to available features
            available_features = [col for col in feature_columns if col in latest.columns]
            features = latest[available_features].fillna(0)
            
            return features
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None
    
    def predict(self, stock_data, sentiment_score=0.0):
        """Make prediction using trained model"""
        try:
            # Get current price
            current_price = float(stock_data['Close'].iloc[-1])
            
            # Try to use the trained model first
            if self.model is not None and self.scaler is not None:
                try:
                    # Prepare features
                    features = self.prepare_features(stock_data, sentiment_score)
                    if features is not None:
                        # Scale features
                        features_scaled = self.scaler.transform(features)
                        
                        # Make prediction
                        prediction = self.model.predict(features_scaled)[0]
                        
                        return {
                            'current_price': round(current_price, 2),
                            'predicted_price': round(prediction, 2),
                            'price_change': round(prediction - current_price, 2),
                            'price_change_pct': round((prediction - current_price) / current_price * 100, 2),
                            'confidence': 0.75,
                            'features_used': len(features.columns),
                            'method': 'ML Model'
                        }
                except Exception as e:
                    print(f"⚠️ ML model failed, using fallback: {e}")
            
            # Fallback: Simple moving average prediction
            print("🔄 Using fallback prediction method...")
            
            # Calculate moving averages
            ma_5 = stock_data['Close'].rolling(5).mean().iloc[-1]
            ma_10 = stock_data['Close'].rolling(10).mean().iloc[-1]
            ma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
            
            # Simple prediction based on trend
            recent_trend = (ma_5 - ma_20) / ma_20
            sentiment_factor = sentiment_score * 0.02  # 2% impact from sentiment
            
            # Predict next price
            predicted_price = current_price * (1 + recent_trend + sentiment_factor)
            
            return {
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'price_change': round(predicted_price - current_price, 2),
                'price_change_pct': round((predicted_price - current_price) / current_price * 100, 2),
                'confidence': 0.60,  # Lower confidence for fallback
                'features_used': 3,
                'method': 'Fallback (Moving Average + Sentiment)'
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None

# Global predictor instance
predictor = ProductionPredictor()

def get_prediction(stock_data, sentiment_score=0.0):
    """Get prediction using the global predictor instance"""
    return predictor.predict(stock_data, sentiment_score)
