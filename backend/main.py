"""
Stock Prediction API with Multiple Data Sources
Uses YFinance, NSEpy, and CSV fallback for Indian stock data
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import time
import random
import yfinance as yf
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
        
        # Load stock data with YFinance as primary source
        print(f"🔮 Making prediction for {stock_symbol}...")
        stock_data = load_stock_data_with_yfinance(stock_symbol, days=60)
        if stock_data is None:
            print(f"❌ No data available for {stock_symbol}")
            return jsonify({'error': f'No data available for {stock_symbol}'}), 404
        
        print(f"✅ Loaded {len(stock_data)} days of data for {stock_symbol}")
        
        # Get sentiment
        sentiment_score = get_news_sentiment(stock_symbol)
        print(f"📰 Sentiment score: {sentiment_score}")
        
        # Use trained ML model for prediction
        print("🤖 Generating prediction with ML model...")
        prediction_result = get_prediction(stock_data, sentiment_score)
        print(f"🎯 Prediction result: {prediction_result}")
        
        if prediction_result is None:
            return jsonify({'error': 'Failed to generate prediction'}), 500
        
        # Extract prediction results
        current_price = prediction_result['current_price']
        predicted_price = prediction_result['predicted_price']
        price_change = prediction_result['price_change']
        price_change_pct = prediction_result['price_change_pct']
        confidence_num = prediction_result['confidence']
        
        # Convert numeric confidence to string
        if confidence_num >= 0.8:
            confidence = "high"
        elif confidence_num >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Get data source info
        data_source = getattr(stock_data, '_data_source', 'Unknown')
        original_symbol = getattr(stock_data, '_original_symbol', stock_symbol)
        
        return jsonify({
            'stock_symbol': original_symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'confidence': confidence,
            'data_source': data_source,
            'sentiment_score': sentiment_score,
            'prediction_method': prediction_result.get('method', 'ML Model'),
            'features_used': prediction_result.get('features_used', 0),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'data_range': {
                'start_date': stock_data.index.min().strftime('%Y-%m-%d'),
                'end_date': stock_data.index.max().strftime('%Y-%m-%d'),
                'days_analyzed': len(stock_data)
            },
            'model_info': {
                'name': prediction_result.get('method', 'ML Model'),
                'rmse': 'N/A',
                'r2': 'N/A',
                'note': f"Using {prediction_result.get('features_used', 0)} features"
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_stock_data_with_yfinance(stock_symbol, days=60):
    """Load stock data using YFinance as primary source with CSV fallback"""
    try:
        # Check cache first
        cached_data = get_cached_data(stock_symbol, days)
        if cached_data is not None:
            return cached_data
        
        print(f"🔄 Loading data for {stock_symbol}...")
        
        # Source 1: Try YFinance API (primary)
        try:
            print("   🔄 Trying YFinance API...")
            yfinance_data = load_yfinance_data(stock_symbol, days)
            if yfinance_data is not None and not yfinance_data.empty:
                yfinance_data._data_source = "YFinance API (live)"
                yfinance_data._original_symbol = stock_symbol
                cache_data(stock_symbol, days, yfinance_data)
                return yfinance_data
            else:
                print(f"   ❌ YFinance API returned no data for {stock_symbol}")
        except Exception as e:
            print(f"   ❌ YFinance API failed: {e}")
        
        # Source 2: Try CSV fallback
        try:
            print("   📁 Using CSV fallback...")
            csv_data = load_csv_fallback(stock_symbol, days)
            print(f"   📁 CSV data result: {csv_data is not None}")
            if csv_data is not None:
                print(f"   📁 CSV data empty: {csv_data.empty}")
                print(f"   📁 CSV data shape: {csv_data.shape if hasattr(csv_data, 'shape') else 'No shape'}")
            
                if csv_data is not None and not csv_data.empty:
                    csv_data._data_source = "CSV (historical)"
                    csv_data._original_symbol = stock_symbol
                    cache_data(stock_symbol, days, csv_data)
                    print(f"   ✅ CSV fallback successful for {stock_symbol}")
                    return csv_data
                else:
                    print(f"   ❌ CSV fallback failed for {stock_symbol} - data is None or empty")
        except Exception as e:
            print(f"   ❌ CSV fallback failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Source 3: Generate realistic data for testing
        try:
            print("   🎲 Generating realistic data for testing...")
            realistic_data = generate_realistic_data(stock_symbol, days)
            if realistic_data is not None and not realistic_data.empty:
                realistic_data._data_source = "Generated (realistic)"
                realistic_data._original_symbol = stock_symbol
                cache_data(stock_symbol, days, realistic_data)
                print(f"   ✅ Generated realistic data for {stock_symbol}")
                return realistic_data
            else:
                print(f"   ❌ Data generation failed for {stock_symbol}")
        except Exception as e:
            print(f"   ❌ Data generation failed: {e}")
        
        print(f"❌ All data sources failed for {stock_symbol}")
        return None
        
    except Exception as e:
        print(f"❌ Error loading data for {stock_symbol}: {e}")
        return None

def load_yfinance_data(symbol, days=60):
    """Load data from YFinance API with proper Indian stock symbols"""
    try:
        print(f"   📊 Fetching {symbol} data from YFinance...")
        
        # Try different symbol formats for Indian stocks
        symbol_formats = [
            f"{symbol}.NS",  # NSE format
            f"{symbol}.BO",  # BSE format
            symbol           # Direct symbol
        ]
        
        for yf_symbol in symbol_formats:
            try:
                print(f"   🔄 Trying symbol: {yf_symbol}")
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=f"{days}d")
                
                if df is not None and not df.empty:
                    # Standardize column names
                    df = df.rename(columns={
                        'Open': 'Open',
                        'High': 'High', 
                        'Low': 'Low',
                        'Close': 'Close',
                        'Volume': 'Volume'
                    })
                    
                    # Ensure we have the required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_columns):
                        print(f"   ✅ YFinance: Loaded {len(df)} days for {symbol} (using {yf_symbol})")
                        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
                        print(f"   📈 Latest Close: ₹{df['Close'].iloc[-1]:.2f}")
                        return df
                    else:
                        print(f"   ❌ YFinance: Missing required columns for {yf_symbol}")
                else:
                    print(f"   ❌ YFinance: No data for {yf_symbol}")
                    
            except Exception as e:
                print(f"   ⚠️ YFinance error for {yf_symbol}: {e}")
                continue
        
        print(f"   ❌ YFinance: No data found for {symbol} in any format")
        return None
            
    except Exception as e:
        print(f"   ❌ YFinance API error: {e}")
        return None

def generate_realistic_data(symbol, days=60):
    """Generate realistic stock data for testing when APIs fail"""
    try:
        print(f"   🎲 Generating realistic data for {symbol}...")
        
        # Base prices for different stocks
        base_prices = {
            'INFY': 1500, 'TCS': 3500, 'RELIANCE': 2500, 'HDFCBANK': 1600,
            'WIPRO': 400, 'BHARTIARTL': 800, 'ITC': 450, 'SBIN': 600,
            'ADANIENT': 3000, 'ADANIPORTS': 1200, 'ASIANPAINT': 3000,
            'AXISBANK': 1000, 'BAJFINANCE': 7000, 'BAJAJFINSV': 1500,
            'COALINDIA': 200, 'DRREDDY': 5000, 'EICHERMOT': 3500,
            'GRASIM': 1500, 'HCLTECH': 1200, 'HDFC': 2500, 'HEROMOTOCO': 2500,
            'HINDALCO': 400, 'HINDUNILVR': 2500, 'ICICIBANK': 900,
            'INDUSINDBK': 1200, 'INFY': 1500, 'JSWSTEEL': 700, 'KOTAKBANK': 1700,
            'LT': 3000, 'M&M': 1200, 'MARUTI': 10000, 'NESTLEIND': 18000,
            'NTPC': 200, 'ONGC': 150, 'POWERGRID': 200, 'SUNPHARMA': 1000,
            'TATAMOTORS': 500, 'TATASTEEL': 100, 'TECHM': 1000, 'TITAN': 3000,
            'ULTRACEMCO': 7000, 'UPL': 600, 'WIPRO': 400
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data with some volatility
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Add some trend and volatility
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change)
            
            # Generate OHLC from close price
            close = current_price
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(100000, 1000000)
            
            prices.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(prices, index=dates)
        
        print(f"   ✅ Generated {len(df)} days of realistic data for {symbol}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"   📈 Latest Close: ₹{df['Close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error generating data for {symbol}: {e}")
        return None

def load_csv_fallback(symbol, days=60):
    """Load data from final_cleaned_data.csv as fallback"""
    try:
        csv_path = "../processed_data/final_cleaned_data.csv"
        print(f"📁 Loading CSV data from {csv_path}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 CSV exists: {os.path.exists(csv_path)}")
        
        if os.path.exists(csv_path):
            # Read the full dataset
            df = pd.read_csv(csv_path)
            print(f"📁 CSV columns: {list(df.columns)}")
            print(f"📁 CSV shape: {df.shape}")
            
            # Filter for the specific stock
            stock_data = df[df['Stock'] == symbol].copy()
            print(f"📁 Filtered data for {symbol}: {len(stock_data)} records")
            
            if len(stock_data) == 0:
                print(f"❌ No data found for {symbol} in CSV")
                return None
            
            # Convert Datetime column to datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Datetime'])
            
            # Select only the OHLCV columns we need
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            stock_data = stock_data[required_columns].copy()
            
            # Set date as index
            stock_data.set_index('Date', inplace=True)
            stock_data.sort_index(inplace=True)
            
            # Get last N days
            if len(stock_data) > days:
                stock_data = stock_data.tail(days)
            
            print(f"✅ Loaded {len(stock_data)} days from CSV for {symbol}")
            print(f"📁 Final DataFrame shape: {stock_data.shape}")
            print(f"📁 Date range: {stock_data.index.min()} to {stock_data.index.max()}")
            return stock_data
        else:
            print(f"❌ CSV not found at {csv_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    try:
        days = request.args.get('days', 100, type=int)
        
        df = load_stock_data_with_yfinance(stock_symbol, days=days)
        if df is None:
            return jsonify({'error': 'Stock not found or no data available'}), 404
        
        data_source = getattr(df, '_data_source', 'Unknown')
        original_symbol = getattr(df, '_original_symbol', stock_symbol)
        
        data = {
            'stock_symbol': original_symbol,
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].tolist(),
            'volumes': df['Volume'].tolist(),
            'timestamp': datetime.now().isoformat(),
            'data_source': data_source,
            'days_requested': days,
            'days_received': len(df)
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_sources': ['YFinance API', 'CSV Files', 'Generated Data'],
        'yfinance_available': True
    })

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction API with YFinance Integration...")
    print("📊 Data Sources: YFinance → CSV")
    print("🇮🇳 Optimized for Indian stock markets (NSE)")
    print("🌐 Starting Flask API server...")
    print("✅ System ready to start")
    print("🔗 Visit http://localhost:5000 for API documentation")
    
    app.run(debug=False, host='0.0.0.0', port=5000)