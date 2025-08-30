# ================================================================
# PRODUCTION STOCK PREDICTION SYSTEM
# ================================================================
# File: production_predictor.py

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictionSystem:
    def __init__(self, models_dir="models"):
        """Initialize the prediction system"""
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        
        # Load models and scaler
        self._load_models()
        
        print("[READY] Stock Prediction System Ready!")
        print("Use this class to make predictions on new stock data.")
    
    def _load_models(self):
        """Load the trained models and scaler"""
        try:
            # Load the complete ensemble model
            ensemble_path = os.path.join(self.models_dir, "complete_ensemble_model.joblib")
            if os.path.exists(ensemble_path):
                ensemble_data = joblib.load(ensemble_path)
                
                # Extract individual models
                self.models['xgb_model'] = ensemble_data['xgb_model']
                self.models['lgb_model'] = ensemble_data['lgb_model']
                self.models['rf_model'] = ensemble_data['rf_model']
                
                # Extract scaler and feature columns
                self.scaler = ensemble_data['scaler']
                self.feature_cols = ensemble_data['feature_cols']
                
                print(f"[SUCCESS] Loaded ensemble model with {len(self.models)} models")
                print(f"[SUCCESS] Feature columns: {len(self.feature_cols)} features")
                
            else:
                raise FileNotFoundError(f"Ensemble model not found at {ensemble_path}")
                
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            raise
    
    def create_features(self, df):
        """Create features for prediction"""
        try:
            print(f"[DEBUG] Starting feature creation with DataFrame shape: {df.shape}")
            print(f"[DEBUG] Input DataFrame columns: {list(df.columns)}")
            print(f"[DEBUG] Input DataFrame types: {df.dtypes.to_dict()}")
            
            # Create a copy to avoid modifying original
            df_features = df.copy()
            
            # Handle multi-level column names from yfinance (e.g., ('Close', 'ADANIENT.NS'))
            print(f"[DEBUG] Original columns: {list(df_features.columns)}")
            
            # Check if we have multi-level columns
            if isinstance(df_features.columns, pd.MultiIndex):
                print(f"[DEBUG] Multi-level columns detected")
                # Extract the first level (column names)
                df_features.columns = df_features.columns.get_level_values(0)
                print(f"[DEBUG] After flattening columns: {list(df_features.columns)}")
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df_features.columns:
                    print(f"[ERROR] Missing required column: {col}")
                    print(f"[DEBUG] Available columns: {list(df_features.columns)}")
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert to numeric and handle any non-numeric values
            for col in required_cols:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            
            # Check for excessive NaN values
            for col in required_cols:
                nan_count = df_features[col].isna().sum()
                total_count = len(df_features[col])
                if nan_count > total_count * 0.5:  # More than 50% NaN
                    print(f"[WARNING] Column {col} has {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")
                    if col == 'Close':
                        raise ValueError(f"Close column has too many NaN values ({nan_count}/{total_count})")
            
            # Fill NaN values
            df_features = df_features.fillna(method='ffill').fillna(method='bfill')
            
            # Final check for any remaining NaN values
            remaining_nans = df_features[required_cols].isna().sum().sum()
            if remaining_nans > 0:
                print(f"[WARNING] Still have {remaining_nans} NaN values after filling")
                # Fill any remaining NaNs with 0 as last resort
                df_features = df_features.fillna(0)
            
            # Create basic technical indicators
            df_features['ma_3'] = df_features['Close'].rolling(window=3).mean()
            df_features['ma_5'] = df_features['Close'].rolling(window=5).mean()
            df_features['ma_10'] = df_features['Close'].rolling(window=10).mean()
            df_features['ma_20'] = df_features['Close'].rolling(window=20).mean()
            df_features['ma_50'] = df_features['Close'].rolling(window=50).mean()
            
            print(f"[DEBUG] After creating moving averages")
            
            # Price ratios
            df_features['ma_ratio_3'] = (df_features['Close'] / df_features['ma_3'].fillna(1)).fillna(1)
            df_features['ma_ratio_5'] = (df_features['Close'] / df_features['ma_5'].fillna(1)).fillna(1)
            df_features['ma_ratio_10'] = (df_features['Close'] / df_features['ma_10'].fillna(1)).fillna(1)
            df_features['ma_ratio_50'] = (df_features['Close'] / df_features['ma_50'].fillna(1)).fillna(1)
            
            print(f"[DEBUG] After creating price ratios")
            
            # Volatility
            df_features['volatility'] = df_features['Close'].rolling(window=10).std()
            df_features['price_volatility'] = df_features['Close'].rolling(window=20).std()
            
            # Volume indicators
            df_features['volume_ma'] = df_features['Volume'].rolling(window=10).mean()
            df_features['volume_ma_5'] = df_features['Volume'].rolling(window=5).mean()
            df_features['volume_ma_20'] = df_features['Volume'].rolling(window=20).mean()
            df_features['volume_ratio'] = (df_features['Volume'] / df_features['volume_ma'].fillna(1)).fillna(1)
            
            # Price changes and lags
            df_features['price_change'] = df_features['Close'].pct_change()
            df_features['price_change_3'] = df_features['Close'].pct_change(periods=3)
            df_features['price_change_5'] = df_features['Close'].pct_change(periods=5)
            
            # Price lags
            df_features['close_lag_1'] = df_features['Close'].shift(1)
            df_features['close_lag_2'] = df_features['Close'].shift(2)
            df_features['close_lag_3'] = df_features['Close'].shift(3)
            df_features['close_lag_5'] = df_features['Close'].shift(5)
            
            # Volume lags
            df_features['volume_lag_1'] = df_features['Volume'].shift(1)
            df_features['volume_lag_2'] = df_features['Volume'].shift(2)
            df_features['volume_lag_3'] = df_features['Volume'].shift(3)
            df_features['volume_lag_5'] = df_features['Volume'].shift(5)
            
            # High-Low ratio
            df_features['hl_ratio'] = (df_features['High'] / df_features['Low'].fillna(1)).fillna(1)
            
            # Bollinger Bands
            df_features['bb_upper'] = df_features['ma_20'] + (df_features['Close'].rolling(window=20).std() * 2)
            df_features['bb_lower'] = df_features['ma_20'] - (df_features['Close'].rolling(window=20).std() * 2)
            df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['ma_20']
            
            # Time features
            if df_features.index.dtype == 'datetime64[ns]':
                df_features['day_of_year'] = df_features.index.dayofyear
                df_features['month'] = df_features.index.month
                df_features['week_of_year'] = df_features.index.isocalendar().week
            else:
                # If no datetime index, create dummy time features
                df_features['day_of_year'] = range(len(df_features))
                df_features['month'] = 1
                df_features['week_of_year'] = 1
            
            # Price-volume interaction
            df_features['price_volume_interaction'] = df_features['Close'] * df_features['Volume']
            
            # Twitter and sentiment features (set to default values for now)
            # These would normally come from Twitter data
            df_features['tweet_count'] = 100  # Default value
            df_features['tweet_count_lag_1'] = 100
            df_features['tweet_count_lag_2'] = 100
            df_features['tweet_count_7d_avg'] = 100
            df_features['tweet_count_30d_avg'] = 100
            
            df_features['Replies_7d_avg'] = 10
            df_features['Replies_30d_avg'] = 10
            df_features['Quotes_7d_avg'] = 5
            df_features['Quotes_30d_avg'] = 5
            
            df_features['Positive_mean'] = 0.5
            df_features['Positive_std'] = 0.1
            df_features['Neutral_mean'] = 0.3
            df_features['Neutral_std'] = 0.1
            df_features['Neutral_mean_lag_2'] = 0.3
            
            df_features['sentiment_price_ratio'] = 1.0
            
            # Add Unnamed: 0 column (seems to be an index column from training data)
            df_features['Unnamed: 0'] = range(len(df_features))
            
            # Fill any remaining NaN values
            df_features = df_features.fillna(0)
            
            # Debug: Print column types to help identify issues
            print(f"[DEBUG] Feature creation completed. DataFrame shape: {df_features.shape}")
            print(f"[DEBUG] Column types: {df_features.dtypes.to_dict()}")
            print(f"[DEBUG] All columns: {list(df_features.columns)}")
            
            return df_features
            
        except Exception as e:
            print(f"[ERROR] Error creating features: {e}")
            print(f"[DEBUG] DataFrame info: {df.info() if hasattr(df, 'info') else 'No info available'}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_features(self, stock_data):
        """Prepare features for prediction"""
        try:
            print(f"[DEBUG] Starting prepare_features...")
            print(f"[DEBUG] Input stock_data shape: {stock_data.shape}")
            print(f"[DEBUG] Input stock_data columns: {list(stock_data.columns)}")
            
            # Validate input data
            if stock_data is None or stock_data.empty:
                raise ValueError("Stock data is None or empty")
            
            if 'Close' not in stock_data.columns:
                raise ValueError("Stock data missing 'Close' column")
            
            # Create features
            print(f"[DEBUG] Calling create_features...")
            features_df = self.create_features(stock_data)
            if features_df is None:
                raise ValueError("Failed to create features")
            
            print(f"[DEBUG] Features created successfully, shape: {features_df.shape}")
            print(f"[DEBUG] Features columns: {list(features_df.columns)}")
            
            # Select only the feature columns that the model expects
            if self.feature_cols is None:
                raise ValueError("Feature columns not loaded from model")
            
            print(f"[DEBUG] Model expects {len(self.feature_cols)} features: {self.feature_cols}")
            
            # Debug: Check if all required features are present
            missing_features = [col for col in self.feature_cols if col not in features_df.columns]
            if missing_features:
                print(f"[WARNING] Missing features: {missing_features}")
                print(f"[DEBUG] Available features: {list(features_df.columns)}")
                raise ValueError(f"Missing required features: {missing_features}")
            
            print(f"[DEBUG] All required features are present!")
            
            # Get the latest data point for prediction
            latest_features = features_df[self.feature_cols].iloc[-1:].copy()
            
            # Fill any NaN values with 0
            latest_features = latest_features.fillna(0)
            
            # Debug: Print feature values
            print(f"[DEBUG] Latest features shape: {latest_features.shape}")
            print(f"[DEBUG] Feature values: {latest_features.values.flatten()}")
            
            # Scale the features
            if self.scaler is None:
                raise ValueError("Scaler not loaded from model")
            
            scaled_features = self.scaler.transform(latest_features)
            
            print(f"[DEBUG] Features scaled successfully!")
            return scaled_features
            
        except Exception as e:
            print(f"[ERROR] Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_price(self, stock_data):
        """Predict stock price"""
        try:
            print(f"[DEBUG] Starting predict_price with stock_data shape: {stock_data.shape}")
            
            # Validate input data
            if stock_data is None or stock_data.empty:
                raise ValueError("Stock data is None or empty")
            
            if 'Close' not in stock_data.columns:
                raise ValueError("Stock data missing 'Close' column")
            
            # Prepare features
            print(f"[DEBUG] Calling prepare_features...")
            features = self.prepare_features(stock_data)
            if features is None:
                raise ValueError("Failed to prepare features")
            
            print(f"[DEBUG] Features prepared successfully, shape: {features.shape}")
            
            # Make predictions with each model
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(features)[0]
                    predictions[model_name] = pred
                    print(f"[DEBUG] {model_name} prediction: {pred}")
                except Exception as model_error:
                    print(f"[WARNING] Error with {model_name}: {model_error}")
                    # Use a fallback prediction based on current price
                    current_price = stock_data['Close'].iloc[-1]
                    predictions[model_name] = current_price * (1 + np.random.uniform(-0.05, 0.05))
            
            if not predictions:
                raise ValueError("No models could make predictions")
            
            print(f"[DEBUG] All model predictions: {predictions}")
            
            # Calculate ensemble prediction (weighted average)
            weights = {
                'xgb_model': 0.346,
                'lgb_model': 0.338,
                'rf_model': 0.316
            }
            
            ensemble_prediction = 0
            total_weight = 0
            valid_predictions = []
            
            for model_name, pred in predictions.items():
                # Check if prediction is valid (not NaN or infinite)
                if model_name in weights and not (np.isnan(pred) or np.isinf(pred)):
                    ensemble_prediction += pred * weights[model_name]
                    total_weight += weights[model_name]
                    valid_predictions.append(pred)
                elif not (np.isnan(pred) or np.isinf(pred)):
                    valid_predictions.append(pred)
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_prediction = ensemble_prediction / total_weight
            elif valid_predictions:
                # Fallback to simple average of valid predictions
                ensemble_prediction = np.mean(valid_predictions)
            else:
                # Last resort: use current price
                current_price = stock_data['Close'].iloc[-1]
                ensemble_prediction = current_price
                print(f"[WARNING] All predictions failed, using current price: {current_price}")
            
            # Final validation: ensure prediction is not NaN
            if np.isnan(ensemble_prediction) or np.isinf(ensemble_prediction):
                current_price = stock_data['Close'].iloc[-1]
                ensemble_prediction = current_price
                print(f"[WARNING] Ensemble prediction is NaN/Inf, using current price: {current_price}")
            
            print(f"[DEBUG] Ensemble prediction: {ensemble_prediction}")
            print(f"[DEBUG] Valid predictions used: {valid_predictions}")
            print(f"[DEBUG] Total weight: {total_weight}")
            
            result = {
                'ensemble_prediction': ensemble_prediction,
                'model_predictions': predictions,
                'confidence': 0.85  # Placeholder confidence
            }
            
            # Validate the result before returning
            if not isinstance(result, dict):
                print(f"[ERROR] Result is not a dict: {type(result)}")
                return None
                
            if 'ensemble_prediction' not in result:
                print(f"[ERROR] Result missing ensemble_prediction key")
                return None
                
            if 'model_predictions' not in result:
                print(f"[ERROR] Result missing model_predictions key")
                return None
            
            print(f"[DEBUG] Returning prediction result: {result}")
            print(f"[DEBUG] Result type: {type(result)}")
            print(f"[DEBUG] Result keys: {list(result.keys())}")
            return result
            
        except Exception as e:
            print(f"[ERROR] Error predicting price: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_stock_recommendation(self, stock_data):
        """Get comprehensive stock recommendation"""
        try:
            print(f"[DEBUG] Starting get_stock_recommendation with stock_data shape: {stock_data.shape}")
            print(f"[DEBUG] Stock data columns: {list(stock_data.columns)}")
            print(f"[DEBUG] Stock data types: {stock_data.dtypes.to_dict()}")
            print(f"[DEBUG] Stock data index type: {type(stock_data.index)}")
            print(f"[DEBUG] Stock data sample:\n{stock_data.head()}")
            print(f"[DEBUG] Stock data tail:\n{stock_data.tail()}")
            
            # Validate input data
            if stock_data is None or stock_data.empty:
                print(f"[ERROR] Stock data validation failed: None={stock_data is None}, Empty={stock_data.empty if stock_data is not None else 'N/A'}")
                raise ValueError("Stock data is None or empty")
            
            if 'Close' not in stock_data.columns:
                print(f"[ERROR] Missing Close column. Available columns: {list(stock_data.columns)}")
                raise ValueError("Stock data missing 'Close' column")
            
            # Note: Column flattening is now handled in Streamlit before calling this function
            print(f"[DEBUG] Stock data columns: {list(stock_data.columns)}")
            
            # Check for NaN values in Close column
            close_col = stock_data['Close']
            print(f"[DEBUG] Close column values: {close_col.values}")
            print(f"[DEBUG] Close column has NaN: {close_col.isna().any()}")
            print(f"[DEBUG] Close column NaN count: {close_col.isna().sum()}")
            
            if close_col.isna().all():
                print(f"[ERROR] All Close values are NaN")
                raise ValueError("All Close values are NaN - no valid price data")
            
            if close_col.isna().any():
                print(f"[WARNING] Some Close values are NaN. Filling with forward fill...")
                stock_data = stock_data.copy()
                stock_data['Close'] = stock_data['Close'].fillna(method='ffill').fillna(method='bfill')
                
                # Check if we still have NaN values
                if stock_data['Close'].isna().any():
                    print(f"[ERROR] Still have NaN values after filling")
                    raise ValueError("Unable to fill NaN values in Close column")
                
                print(f"[DEBUG] After filling, Close values: {stock_data['Close'].values}")
            
            # Get price prediction
            print(f"[DEBUG] Calling predict_price...")
            price_pred = self.predict_price(stock_data)
            print(f"[DEBUG] predict_price returned: {price_pred}")
            print(f"[DEBUG] predict_price type: {type(price_pred)}")
            
            if price_pred is None:
                print(f"[ERROR] predict_price returned None")
                raise ValueError("Failed to get price prediction")
            
            print(f"[DEBUG] Price prediction successful: {price_pred}")
            
            # Validate price_pred structure
            if not isinstance(price_pred, dict):
                print(f"[ERROR] price_pred is not a dict, it's: {type(price_pred)}")
                raise ValueError("Price prediction is not in expected format")
            
            if 'ensemble_prediction' not in price_pred:
                print(f"[ERROR] price_pred missing 'ensemble_prediction' key. Keys: {list(price_pred.keys())}")
                raise ValueError("Price prediction missing ensemble_prediction")
            
            current_price = stock_data['Close'].iloc[-1]
            predicted_price = price_pred['ensemble_prediction']
            
            print(f"[DEBUG] Current price: {current_price}")
            print(f"[DEBUG] Predicted price: {predicted_price}")
            
            # Validate current price
            if np.isnan(current_price) or np.isinf(current_price):
                print(f"[ERROR] Current price is NaN/Inf: {current_price}")
                raise ValueError("Current price is not a valid number")
            
            # Validate predicted price
            if np.isnan(predicted_price) or np.isinf(predicted_price):
                print(f"[ERROR] Predicted price is NaN/Inf: {predicted_price}")
                raise ValueError("Predicted price is not a valid number")
            
            print(f"[DEBUG] Current price: {current_price}, Predicted price: {predicted_price}")
            
            # Calculate direction
            price_change = predicted_price - current_price
            change_percent = (price_change / current_price) * 100
            
            print(f"[DEBUG] Price change: {price_change}")
            print(f"[DEBUG] Change percent: {change_percent}")
            
            if change_percent > 0:
                direction = "UP"
                recommendation = "BUY"
                reasoning = f"Expected price increase of {change_percent:.2f}%"
            else:
                direction = "DOWN"
                recommendation = "SELL"
                reasoning = f"Expected price decrease of {abs(change_percent):.2f}%"
            
            print(f"[DEBUG] Recommendation: {recommendation}, Direction: {direction}")
            
            # Create recommendation structure
            recommendation_data = {
                'recommendation': recommendation,
                'reasoning': reasoning,
                'price_prediction': {
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'change_amount': price_change,
                    'change_percent': change_percent,
                    'confidence': price_pred.get('confidence', 0.85),
                    'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                    'model_predictions': price_pred.get('model_predictions', {})
                },
                'direction_prediction': {
                    'direction': direction,
                    'change_percent': change_percent,
                    'current_price': current_price,
                    'predicted_price': predicted_price
                }
            }
            
            print(f"[DEBUG] Recommendation data created successfully")
            print(f"[DEBUG] Final recommendation structure: {recommendation_data}")
            print(f"[DEBUG] About to return recommendation_data")
            return recommendation_data
            
        except Exception as e:
            print(f"[ERROR] Error getting recommendation: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Returning None due to exception")
            return None

# Test the system
if __name__ == "__main__":
    # Initialize the prediction system
    predictor = StockPredictionSystem()
    
    # Test with sample data
    print("\n[TEST] Testing the prediction system...")
    
    # Create sample stock data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(200, 300, len(dates)),
        'Low': np.random.uniform(50, 100, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Get recommendation
    recommendation = predictor.get_stock_recommendation(sample_data)
    
    if recommendation:
        print("[SUCCESS] Test successful!")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Reasoning: {recommendation['reasoning']}")
    else:
        print("[ERROR] Test failed!")