from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import pandas as pd
import numpy as np  
import time
import random
import warnings
from smartapi import SmartConnect

# Angel One API credentials (keep secret in production!)

ANGEL_API_KEY = "OWZNREVh"
ANGEL_API_SECRET = "555517ea-2a99-491c-a791-9d0b6ff04fbc"
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_CLIENT_PWD = os.environ.get("ANGEL_CLIENT_PWD", "")
ANGEL_TOTP = os.environ.get("ANGEL_TOTP", "")
angel_obj = None


def get_angel_one_session():
    global angel_obj
    if angel_obj is not None:
        return angel_obj
    angel_obj = SmartConnect(api_key=ANGEL_API_KEY)
    try:
        angel_obj.generateSession(ANGEL_CLIENT_ID, ANGEL_CLIENT_PWD, ANGEL_TOTP)
    except Exception as e:
        print(f"Angel One login failed: {e}")
        angel_obj = None
    return angel_obj


def get_angel_one_ltp(symbol):
    obj = get_angel_one_session()
    if obj is None:
        return None
    symbol_map = {
        'RELIANCE': '2885', 'INFY': '1594', 'TCS': '11536', 'HDFCBANK': '1333',
        'ICICIBANK': '4963', 'SBIN': '3045', 'WIPRO': '3787', 'HCLTECH': '7229',
        'ITC': '1660', 'BHARTIARTL': '2712', 'TATASTEEL': '3499', 'HDFC': '1330'
    }
    token = symbol_map.get(symbol.upper())
    if not token:
        print(f"No Angel One token for symbol: {symbol}")
        return None
    try:
        params = {"exchange": "NSE", "tradingsymbol": f"{symbol.upper()}-EQ", "symboltoken": token}
        ltp_data = obj.ltpData(**params)
        price = ltp_data['data']['ltp']
        return price
    except Exception as e:
        print(f"Angel One LTP fetch failed for {symbol}: {e}")
        return None


app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this'

@app.route('/')
def index():
    return jsonify({
        'message': 'Stock Prediction API (Angel One Only)',
        'version': '5.0',
        'features': [
            'Angel One SmartAPI for live Indian stock data',
            'No fallback, no yfinance, no CSV'
        ],
        'endpoints': {
            'predict': '/predict',
            'stock_data': '/stock_data/<symbol>',
            'health': '/health'
        },
        'data_source': 'Angel One SmartAPI',
        'optimized_for': 'Indian stock markets (NSE)',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    stock_symbol = data.get('stock_symbol', 'INFY')
    days = data.get('days', 1)  # Default to 1 if not provided
    ltp = get_angel_one_ltp(stock_symbol)
    if ltp is None:
        return jsonify({'error': f'No data available for {stock_symbol}'}), 404
    # ...existing code...
    return jsonify({
        'stock_symbol': stock_symbol,
        'current_price': ltp,
        'data_source': 'Angel One SmartAPI',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stock_data/<stock_symbol>', methods=['GET'])
def get_stock_data(stock_symbol):
    ltp = get_angel_one_ltp(stock_symbol)
    if ltp is None:
        return jsonify({'error': 'Stock not found or no data available'}), 404
    return jsonify({
        'stock_symbol': stock_symbol,
        'current_price': ltp,
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Angel One SmartAPI'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Angel One SmartAPI'
    })

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction API with Angel One Integration...")
    print("🇮🇳 Optimized for Indian stock markets (NSE)")
    print("🌐 Starting Flask API server...")
    print("✅ System ready to start")
    print("🔗 Visit http://localhost:5000 for API documentation")
    app.run(debug=False, host='0.0.0.0', port=5000)
from production_predictor import get_prediction

# Disable warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this'

# Cache configuration
CACHE_DURATION = 30  # 30 seconds for development
stock_data_cache = {}

@app.route('/')
def index():
    """API root endpoint with NSEpy integration info"""
    return jsonify({
        'message': 'Stock Prediction API with Multiple Data Sources',
        'version': '4.0',
        'features': [
            'YFinance for live Indian stock data',
            'CSV fallback for historical data',
            'ML-based price predictions',
            'No authentication required'
        ],
        'endpoints': {
            'predict': '/predict',
            'stocks': '/stocks',
            'stock_data': '/stock_data/<symbol>',
            'health': '/health'
        },
        'data_sources': ['YFinance API', 'CSV Files', 'Generated Data'],
        'optimized_for': 'Indian stock markets (NSE)',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make stock price prediction using NSEpy and CSV data sources"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        stock_symbol = data.get('stock_symbol', 'INFY')
        days = data.get('days', 1)  # Default to 1 if not provided
        
        # Load stock data with YFinance as primary source
        print(f"� Loading data for {stock_symbol}...")
        # Only use Angel One SmartAPI for LTP (latest price only)
        try:
            print("   🟢 Trying Angel One SmartAPI for LTP...")
            ltp = get_angel_one_ltp(stock_symbol)
            if ltp is not None:
                # Create a minimal DataFrame with just the latest price
                now = datetime.now()
                df = pd.DataFrame({
                    'Open': [ltp], 'High': [ltp], 'Low': [ltp], 'Close': [ltp], 'Volume': [0]
                }, index=[now])
                df._data_source = "Angel One SmartAPI (live LTP)"
                df._original_symbol = stock_symbol
                cache_data(stock_symbol, days, df)
                print(f"   ✅ Angel One LTP: ₹{ltp} for {stock_symbol}")
                return df
            else:
                print(f"   ❌ Angel One LTP not available for {stock_symbol}")
        except Exception as e:
            print(f"   ❌ Angel One LTP failed: {e}")
        print(f"❌ All data sources failed for {stock_symbol}")
        return None
        
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_cached_data(stock_symbol, days):
    """Get cached data if available and fresh"""
    cache_key = f"{stock_symbol}_{days}"
    
    if cache_key in stock_data_cache:
        cached_data, timestamp = stock_data_cache[cache_key]
        if time.time() - timestamp < CACHE_DURATION:
            print(f"📋 Using cached data for {stock_symbol}")
            return cached_data
        else:
            del stock_data_cache[cache_key]
    
    return None

def cache_data(stock_symbol, days, data):
    """Cache stock data"""
    cache_key = f"{stock_symbol}_{days}"
    stock_data_cache[cache_key] = (data, time.time())

def get_news_sentiment(stock_symbol):
    """Get news sentiment score (simplified)"""
    try:
        # Simple sentiment calculation based on stock symbol
        sentiment = random.uniform(-0.1, 0.1)  # Random sentiment between -10% and +10%
        print(f"📰 News sentiment for {stock_symbol}: {sentiment:.3f}")
        return sentiment
    except:
        return 0.0

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the stock data cache"""
    try:
        global stock_data_cache
        stock_data_cache.clear()
        return jsonify({
            'message': 'Cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stocks', methods=['GET'])
def get_available_stocks():
    """Get list of available stocks"""
    try:
        # Popular Indian stocks available on NSE
        indian_stocks = [
            'INFY', 'TCS', 'RELIANCE', 'HDFC', 'HDFCBANK', 
            'ICICIBANK', 'SBIN', 'WIPRO', 'BHARTIARTL', 'ITC',
            'HCLTECH', 'ADANIENT', 'TATASTEEL', 'BAJFINANCE',
            'KOTAKBANK', 'ASIANPAINT', 'MARUTI', 'NESTLEIND',
            'ULTRACEMCO', 'TITAN', 'POWERGRID', 'NTPC'
        ]
        
        return jsonify({
            'stocks': indian_stocks,
            'count': len(indian_stocks),
            'timestamp': datetime.now().isoformat(),
            'note': 'Indian stocks available via NSEpy API and CSV fallback'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stock_data/<stock_symbol>', methods=['GET'])
def get_stock_data(stock_symbol):
    """Get stock data for visualization"""
    ltp = get_angel_one_ltp(stock_symbol)
    if ltp is None:
        return jsonify({'error': 'Stock not found or no data available'}), 404
    return jsonify({
        'stock_symbol': stock_symbol,
        'current_price': ltp,
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Angel One SmartAPI'
    })
   

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction API with YFinance Integration...")
    print("📊 Data Sources: YFinance → CSV")
    print("🇮🇳 Optimized for Indian stock markets (NSE)")
    print("🌐 Starting Flask API server...")
    print("✅ System ready to start")
    print("🔗 Visit http://localhost:5000 for API documentation")
    
    app.run(debug=False, host='0.0.0.0', port=5000)

# All code below this line is removed as it is unreachable, duplicate, or unused.