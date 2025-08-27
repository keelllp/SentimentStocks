"""
Flask backend for Stock Price Prediction Web Application
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for loaded models
models = {}
scaler = None
feature_names = None

def load_models():
    """Load trained models and scaler"""
    global models, scaler, feature_names
    
    models_dir = "../models"
    
    try:
        # Load XGBoost model
        xgb_path = os.path.join(models_dir, "stock_predictor_xgboost.pkl")
        if os.path.exists(xgb_path):
            models['xgboost'] = joblib.load(xgb_path)
            print("XGBoost model loaded successfully")
        
        # Load LightGBM model
        lgb_path = os.path.join(models_dir, "stock_predictor_lightgbm.pkl")
        if os.path.exists(lgb_path):
            models['lightgbm'] = joblib.load(lgb_path)
            print("LightGBM model loaded successfully")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, "stock_predictor_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
        
        # Load feature names
        features_path = os.path.join(models_dir, "stock_predictor_features.pkl")
        if os.path.exists(features_path):
            feature_names = joblib.load(features_path)
            print("Feature names loaded successfully")
        
        return len(models) > 0 and scaler is not None and feature_names is not None
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make stock price prediction"""
    try:
        data = request.get_json()
        stock_symbol = data.get('stock_symbol', 'INFY')
        model_type = data.get('model_type', 'ensemble')
        
        # Check if models are loaded
        if not models or scaler is None or feature_names is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Load latest stock data
        stock_data = load_latest_stock_data(stock_symbol)
        if stock_data is None:
            return jsonify({'error': f'No data available for {stock_symbol}'}), 404
        
        # Prepare features
        features = prepare_features(stock_data)
        if features is None:
            return jsonify({'error': 'Feature preparation failed'}), 500
        
        # Make prediction
        prediction = make_prediction(features, model_type)
        
        # Get current price
        current_price = stock_data['Close'].iloc[-1]
        
        # Calculate change
        price_change = prediction - current_price
        price_change_pct = (price_change / current_price) * 100
        
        result = {
            'stock_symbol': stock_symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(prediction, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'model_used': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_latest_stock_data(stock_symbol):
    """Load latest stock data for prediction"""
    try:
        stock_path = f"../data/stock_data/{stock_symbol}.csv"
        if not os.path.exists(stock_path):
            return None
        
        # Load last 30 days of data for feature calculation
        df = pd.read_csv(stock_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Get last 30 days
        return df.tail(30)
        
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def prepare_features(stock_data):
    """Prepare features for prediction"""
    try:
        # This is a simplified feature preparation
        # In a real application, you would use the same feature engineering pipeline
        
        features = {}
        
        # Basic price features
        features['Close'] = stock_data['Close'].iloc[-1]
        features['Open'] = stock_data['Open'].iloc[-1]
        features['High'] = stock_data['High'].iloc[-1]
        features['Low'] = stock_data['Low'].iloc[-1]
        
        # Moving averages
        features['ma_5'] = stock_data['Close'].rolling(5).mean().iloc[-1]
        features['ma_20'] = stock_data['Close'].rolling(20).mean().iloc[-1]
        
        # Price changes
        features['price_change_1d'] = stock_data['Close'].pct_change().iloc[-1]
        features['price_change_5d'] = stock_data['Close'].pct_change(5).iloc[-1]
        
        # Volatility
        features['volatility_5d'] = stock_data['Close'].pct_change().rolling(5).std().iloc[-1]
        
        # Volume features (if available)
        if 'Volume' in stock_data.columns:
            features['volume'] = stock_data['Volume'].iloc[-1]
            features['volume_ma_5'] = stock_data['Volume'].rolling(5).mean().iloc[-1]
        else:
            features['volume'] = 1000  # Default value
            features['volume_ma_5'] = 1000
        
        # Time features
        today = datetime.now()
        features['day_of_week'] = today.weekday()
        features['month'] = today.month
        features['quarter'] = (today.month - 1) // 3 + 1
        
        # Create feature vector
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        return np.array(feature_vector).reshape(1, -1)
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None

def make_prediction(features, model_type):
    """Make prediction using loaded models"""
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        if model_type == 'xgboost' and 'xgboost' in models:
            prediction = models['xgboost'].predict(features_scaled)[0]
        elif model_type == 'lightgbm' and 'lightgbm' in models:
            prediction = models['lightgbm'].predict(features_scaled)[0]
        elif model_type == 'ensemble' and len(models) > 1:
            # Ensemble prediction
            predictions = []
            for model in models.values():
                pred = model.predict(features_scaled)[0]
                predictions.append(pred)
            prediction = np.mean(predictions)
        else:
            # Use first available model
            model_name = list(models.keys())[0]
            prediction = models[model_name].predict(features_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative price
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

@app.route('/api/stocks')
def get_available_stocks():
    """Get list of available stocks"""
    try:
        stock_dir = "../data/stock_data"
        if not os.path.exists(stock_dir):
            return jsonify([])
        
        stock_files = [f.replace('.csv', '') for f in os.listdir(stock_dir) 
                      if f.endswith('.csv')]
        
        return jsonify(stock_files)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_data/<stock_symbol>')
def get_stock_data(stock_symbol):
    """Get stock data for visualization"""
    try:
        stock_path = f"../data/stock_data/{stock_symbol}.csv"
        if not os.path.exists(stock_path):
            return jsonify({'error': 'Stock not found'}), 404
        
        df = pd.read_csv(stock_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get last 100 days
        df = df.tail(100)
        
        data = {
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].tolist(),
            'volumes': df['Volume'].tolist() if 'Volume' in df.columns else [1000] * len(df)
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = len(models) > 0 and scaler is not None and feature_names is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': len(models),
        'scaler_loaded': scaler is not None,
        'features_loaded': feature_names is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Loading models...")
    if load_models():
        print("Models loaded successfully!")
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please ensure models are trained first.")
        print("You can still run the application, but predictions will not work.")
        app.run(debug=True, host='0.0.0.0', port=5000)
