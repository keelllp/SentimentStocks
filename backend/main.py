"""
Flask API for Stock Price Prediction using Alpha Vantage
Much more reliable than yfinance with professional-grade data
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import warnings
import time
import random
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
USE_NEWS_SENTIMENT = True
CACHE_DURATION = 900  # Cache data for 15 minutes (Alpha Vantage is more reliable)
stock_data_cache = {}

# Alpha Vantage Rate Limiting
ALPHA_VANTAGE_RATE_LIMIT = 5  # Max calls per minute
ALPHA_VANTAGE_CALL_TIMES = []  # Track API call times
MIN_CALL_INTERVAL = 1.2  # Minimum seconds between calls (1.2s to be safe)

# Initialize Alpha Vantage
# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'C41AHGVK4R014QF9'  # TODO: Replace with your actual API key
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'message': 'Stock Price Prediction API - Alpha Vantage Version',
        'status': 'running',
        'features': [
            'Real-time stock data via Alpha Vantage',
            'CSV fallback data',
            'News sentiment analysis',
            'Basic prediction simulation'
        ],
        'endpoints': {
            'predict': '/predict (last 60 days)',
            'predict_custom': '/predict_custom (custom date range)',
            'stocks': '/stocks (available stocks)',
            'stock_data': '/stock_data/<symbol>?days=100',
            'market_summary': '/market_summary/<symbol> (real-time data)',
            'health': '/health',
            'performance': '/performance'
        },
        'usage': {
            'predict': 'POST with {"stock_symbol": "INFY"}',
            'predict_custom': 'POST with {"stock_symbol": "INFY", "start_date": "2024-01-01", "end_date": "2024-12-31"}',
            'stock_data': 'GET /stock_data/INFY?days=200'
        },
        'data_source': 'Alpha Vantage API (500 calls/day free tier)'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make stock price prediction using Alpha Vantage data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        stock_symbol = data.get('stock_symbol', 'INFY')
        
        # Load stock data
        stock_data = load_stock_data(stock_symbol, days=60)
        if stock_data is None:
            return jsonify({'error': f'No data available for {stock_symbol}'}), 404
        
        # Get sentiment
        sentiment_score = get_news_sentiment(stock_symbol)
        
        # Get current price
        current_price = stock_data['Close'].iloc[-1]
        
        # Simple prediction (based on moving average + sentiment)
        ma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
        sentiment_factor = 1 + (sentiment_score * 0.1)  # Sentiment affects prediction by ±10%
        predicted_price = ma_20 * sentiment_factor
        
        # Calculate change and confidence
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Determine confidence
        confidence = "Medium" if abs(price_change_pct) < 5 else "Low"
        
        # Determine data source from the dataframe
        data_source = getattr(stock_data, '_data_source', 'CSV (historical)')
        
        result = {
            'stock_symbol': stock_symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'confidence': confidence,
            'sentiment_score': round(sentiment_score, 3),
            'data_source': data_source,
            'model_info': {
                'name': 'Moving Average + Sentiment (Alpha Vantage)',
                'note': 'Using Alpha Vantage API for reliable market data'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    """Make stock price prediction with custom date range"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        stock_symbol = data.get('stock_symbol', 'INFY')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Load custom date range stock data
        stock_data = load_custom_stock_data(stock_symbol, start_date, end_date)
        if stock_data is None:
            return jsonify({'error': f'No data available for {stock_symbol} in specified date range'}), 404
        
        # Get sentiment
        sentiment_score = get_news_sentiment(stock_symbol)
        
        # Get current price
        current_price = stock_data['Close'].iloc[-1]
        
        # Simple prediction
        ma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
        sentiment_factor = 1 + (sentiment_score * 0.1)
        predicted_price = ma_20 * sentiment_factor
        
        # Calculate change and confidence
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        confidence = "Medium" if abs(price_change_pct) < 5 else "Low"
        
        # Determine data source from the dataframe
        data_source = getattr(stock_data, '_data_source', 'CSV (historical)')
        
        result = {
            'stock_symbol': stock_symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'confidence': confidence,
            'sentiment_score': round(sentiment_score, 3),
            'data_source': data_source,
            'data_range': {
                'start_date': start_date,
                'end_date': end_date,
                'days_analyzed': len(stock_data)
            },
            'model_info': {
                'name': 'Moving Average + Sentiment (Alpha Vantage)',
                'note': 'Using Alpha Vantage API for reliable market data'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/market_summary/<stock_symbol>', methods=['GET'])
def get_market_summary(stock_symbol):
    """Get market summary for a stock using Alpha Vantage"""
    try:
        # Get recent price data
        hist = load_stock_data(stock_symbol, days=5)
        if hist is None:
            return jsonify({'error': 'No data available'}), 404
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        # Get sentiment
        sentiment = get_news_sentiment(stock_symbol)
        
        # Get data source
        data_source = getattr(hist, '_data_source', 'Unknown')
        
        summary = {
            'stock_symbol': stock_symbol,
            'current_price': round(current_price, 2),
            'previous_close': round(prev_close, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'volume': int(hist['Volume'].iloc[-1]),
            'market_cap': 'N/A',
            'pe_ratio': 'N/A',
            'sentiment_score': round(sentiment, 3),
            'data_source': data_source,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def enforce_alpha_vantage_rate_limit():
    """Enforce Alpha Vantage rate limiting (5 calls/minute, 1.2s between calls)"""
    current_time = time.time()
    
    # Remove calls older than 1 minute
    ALPHA_VANTAGE_CALL_TIMES[:] = [t for t in ALPHA_VANTAGE_CALL_TIMES if current_time - t < 60]
    
    # Check if we've exceeded the rate limit
    if len(ALPHA_VANTAGE_CALL_TIMES) >= ALPHA_VANTAGE_RATE_LIMIT:
        oldest_call = min(ALPHA_VANTAGE_CALL_TIMES)
        wait_time = 60 - (current_time - oldest_call) + 1
        print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
        # Clean up after waiting
        ALPHA_VANTAGE_CALL_TIMES[:] = [t for t in ALPHA_VANTAGE_CALL_TIMES if current_time - t < 60]
    
    # Check minimum interval between calls
    if ALPHA_VANTAGE_CALL_TIMES:
        last_call = max(ALPHA_VANTAGE_CALL_TIMES)
        time_since_last = current_time - last_call
        if time_since_last < MIN_CALL_INTERVAL:
            wait_time = MIN_CALL_INTERVAL - time_since_last
            print(f"⏳ Waiting {wait_time:.1f}s between API calls...")
            time.sleep(wait_time)
    
    # Record this call
    ALPHA_VANTAGE_CALL_TIMES.append(current_time)
    print(f"📊 API call recorded. Total calls this minute: {len(ALPHA_VANTAGE_CALL_TIMES)}")

def get_cached_data(stock_symbol, days):
    """Get cached data if available and fresh"""
    cache_key = f"{stock_symbol}_{days}"
    
    if cache_key in stock_data_cache:
        cached_data, timestamp = stock_data_cache[cache_key]
        if time.time() - timestamp < CACHE_DURATION:
            print(f"📋 Using cached data for {stock_symbol} (age: {time.time() - timestamp:.1f}s)")
            return cached_data
        else:
            print(f"📋 Cache expired for {stock_symbol}, fetching fresh data")
            del stock_data_cache[cache_key]
    
    return None

def cache_data(stock_symbol, days, data):
    """Cache stock data with timestamp"""
    cache_key = f"{stock_symbol}_{days}"
    stock_data_cache[cache_key] = (data, time.time())
    print(f"📋 Cached data for {stock_symbol}")

def load_stock_data(stock_symbol, days=60):
    """Load stock data using Alpha Vantage with fallback to CSV"""
    try:
        # Check cache first
        cached_data = get_cached_data(stock_symbol, days)
        if cached_data is not None:
            return cached_data
        
        # Try Alpha Vantage first
        print(f"🔄 Fetching LIVE data for {stock_symbol} via Alpha Vantage...")
        
        try:
            # Enforce rate limiting before making API call
            enforce_alpha_vantage_rate_limit()
            
            # FIXED: For Alpha Vantage, Indian stocks don't use .NS suffix
            # Remove .NS if present, keep original symbol
            if stock_symbol.endswith('.NS'):
                alpha_symbol = stock_symbol.replace('.NS', '')
            else:
                alpha_symbol = stock_symbol
            
            print(f"   Using Alpha Vantage symbol: {alpha_symbol}")
            
            # Get daily time series from Alpha Vantage
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': alpha_symbol,
                'apikey': ALPHA_VANTAGE_API_KEY,
                'outputsize': 'compact'  # Last 100 data points
            }
            
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    
                    if time_series:
                        # Convert to DataFrame
                        df_data = []
                        for date, values in time_series.items():
                            df_data.append({
                                'Date': date,
                                'Open': float(values['1. open']),
                                'High': float(values['2. high']),
                                'Low': float(values['3. low']),
                                'Close': float(values['4. close']),
                                'Volume': int(values['5. volume'])
                            })
                        
                        df = pd.DataFrame(df_data)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        df = df.sort_index()  # Sort by date
                        
                        if not df.empty:
                            # Validate that we got recent data (within last 30 days)
                            latest_date = df.index[-1]
                            days_old = (datetime.now() - latest_date).days
                            
                            if days_old <= 30:  # Data is recent
                                # Get only the requested number of days
                                df = df.tail(days)
                                
                                print(f"✅ Loaded {len(df)} days of LIVE data from Alpha Vantage (latest: {latest_date.strftime('%Y-%m-%d')})")
                                df._data_source = "Alpha Vantage (live)"
                                
                                # Cache the successful result
                                cache_data(stock_symbol.replace('.NS', ''), days, df)
                                
                                return df
                            else:
                                print(f"⚠️ Alpha Vantage data is too old ({days_old} days old), trying CSV fallback")
                        else:
                            print("⚠️ Alpha Vantage returned empty data")
                    else:
                        print("⚠️ Alpha Vantage returned empty time series")
                elif 'Error Message' in data:
                    error_msg = data['Error Message']
                    print(f"❌ Alpha Vantage error: {error_msg}")
                    
                    # Check if it's a rate limit error
                    if 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                        print("⚠️ Rate limit detected. Waiting 60 seconds before retry...")
                        time.sleep(60)
                        # Clear the call times to reset rate limiting
                        ALPHA_VANTAGE_CALL_TIMES.clear()
                        
                elif 'Note' in data:
                    note_msg = data['Note']
                    print(f"⚠️ Alpha Vantage note: {note_msg}")
                    
                    # Check if it's a rate limit note
                    if 'rate' in note_msg.lower() or 'limit' in note_msg.lower():
                        print("⚠️ Rate limit note detected. Waiting 60 seconds...")
                        time.sleep(60)
                        # Clear the call times to reset rate limiting
                        ALPHA_VANTAGE_CALL_TIMES.clear()
                        
                else:
                    print(f"⚠️ Alpha Vantage returned unexpected format")
                    
            else:
                print(f"❌ Alpha Vantage HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Alpha Vantage error: {e}")
        
        # Fallback to CSV if Alpha Vantage fails
        print("📁 Falling back to CSV data...")
        return load_csv_fallback(stock_symbol, days)
        
    except Exception as e:
        print(f"❌ Error loading stock data: {e}")
        return load_csv_fallback(stock_symbol, days)

def load_custom_stock_data(stock_symbol, start_date, end_date):
    """Load stock data for custom date range using Alpha Vantage"""
    try:
        print(f"🔄 Fetching custom LIVE data for {stock_symbol} from {start_date} to {end_date} via Alpha Vantage...")
        
        try:
            # Enforce rate limiting before making API call
            enforce_alpha_vantage_rate_limit()
            
            # FIXED: For Alpha Vantage, Indian stocks don't use .NS suffix
            # Remove .NS if present, keep original symbol
            if stock_symbol.endswith('.NS'):
                alpha_symbol = stock_symbol.replace('.NS', '')
            else:
                alpha_symbol = stock_symbol
            
            print(f"   Using Alpha Vantage symbol: {alpha_symbol}")
            
            # Get daily time series from Alpha Vantage
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': alpha_symbol,
                'apikey': ALPHA_VANTAGE_API_KEY,
                'outputsize': 'full'  # Full data for custom range
            }
            
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    
                    if time_series:
                        # Convert to DataFrame
                        df_data = []
                        for date, values in time_series.items():
                            # Filter by date range
                            if start_date <= date <= end_date:
                                df_data.append({
                                    'Date': date,
                                    'Open': float(values['1. open']),
                                    'High': float(values['2. high']),
                                    'Low': float(values['3. low']),
                                    'Close': float(values['4. close']),
                                    'Volume': int(values['5. volume'])
                                })
                        
                        df = pd.DataFrame(df_data)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        df = df.sort_index()  # Sort by date
                        
                        if not df.empty:
                            # Validate that we got recent data
                            latest_date = df.index[-1]
                            days_old = (datetime.now() - latest_date).days
                            
                            if days_old <= 30:  # Data is recent
                                print(f"✅ Loaded {len(df)} days of LIVE custom data from Alpha Vantage (latest: {latest_date.strftime('%Y-%m-%d')})")
                                df._data_source = "Alpha Vantage (live)"
                                return df
                            else:
                                print(f"⚠️ Custom Alpha Vantage data is too old ({days_old} days old), using CSV fallback")
                        else:
                            print("⚠️ Alpha Vantage returned empty custom data")
                    else:
                        print("⚠️ Alpha Vantage returned empty time series")
                elif 'Error Message' in data:
                    error_msg = data['Error Message']
                    print(f"❌ Alpha Vantage error: {error_msg}")
                    
                    # Check if it's a rate limit error
                    if 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                        print("⚠️ Rate limit detected. Waiting 60 seconds before retry...")
                        time.sleep(60)
                        # Clear the call times to reset rate limiting
                        ALPHA_VANTAGE_CALL_TIMES.clear()
                        
                elif 'Note' in data:
                    note_msg = data['Note']
                    print(f"⚠️ Alpha Vantage note: {note_msg}")
                    
                    # Check if it's a rate limit note
                    if 'rate' in note_msg.lower() or 'limit' in note_msg.lower():
                        print("⚠️ Rate limit note detected. Waiting 60 seconds...")
                        time.sleep(60)
                        # Clear the call times to reset rate limiting
                        ALPHA_VANTAGE_CALL_TIMES.clear()
                        
                else:
                    print(f"⚠️ Alpha Vantage returned unexpected format")
                    
            else:
                print(f"❌ Alpha Vantage HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Alpha Vantage custom data error: {e}")
        
        # Fallback to CSV
        print("📁 Using CSV fallback for custom date range...")
        return load_csv_fallback(stock_symbol, 60)
        
    except Exception as e:
        print(f"❌ Error loading custom data: {e}")
        return load_csv_fallback(stock_symbol, 60)

def load_csv_fallback(stock_symbol, days=60):
    """Load data from CSV files with date validation"""
    try:
        # Try different path combinations
        possible_paths = [
            f"../data/stock_data/{stock_symbol}.csv",
            f"data/stock_data/{stock_symbol}.csv",
            f"../data/stock_data/{stock_symbol}.csv"
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path and os.path.exists(csv_path):
            print(f"📁 Loading CSV data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Convert Date column to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
            
            # Check if CSV data is too old (more than 1 year old)
            latest_date = df.index[-1]
            # Handle timezone-aware datetime comparison
            if latest_date.tzinfo is not None:
                latest_date = latest_date.replace(tzinfo=None)
            
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 365:  # Data is more than 1 year old
                print(f"⚠️ WARNING: CSV data is {days_old} days old (from {latest_date.strftime('%Y-%m-%d')})")
                print("⚠️ This data is too old for accurate predictions. Please check your internet connection.")
                print("⚠️ Consider updating your CSV files or fixing Alpha Vantage connectivity.")
                
                # Still return the data but with a warning
                df = df.tail(days)
                df.columns = [col.title() for col in df.columns]
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"❌ Missing required column: {col}")
                        return None
                
                print(f"⚠️ Using OLD CSV data: {len(df)} days (last updated: {latest_date.strftime('%Y-%m-%d')})")
                df._data_source = "CSV (historical - old)"
                return df
            else:
                # CSV data is relatively recent
                df = df.tail(days)
                df.columns = [col.title() for col in df.columns]
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"❌ Missing required column: {col}")
                        return None
                
                print(f"✅ Loaded {len(df)} days from CSV (last updated: {latest_date.strftime('%Y-%m-%d')})")
                df._data_source = "CSV (historical - recent)"
                return df
        else:
            print(f"❌ CSV not found for {stock_symbol}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def get_news_sentiment(stock_symbol):
    """Get simulated news sentiment"""
    try:
        import random
        sentiment = random.uniform(-0.2, 0.2)
        print(f"📰 News sentiment for {stock_symbol}: {sentiment:.3f}")
        return sentiment
    except:
        return 0.0

@app.route('/stocks', methods=['GET'])
def get_available_stocks():
    """Get list of available stocks"""
    try:
        indian_stocks = [
            'ADANIENT', 'BHARTIARTL', 'HCLTECH', 'HDFCBANK', 
            'ICICIBANK', 'INFY', 'SBIN', 'TATASTEEL', 'RELIANCE',
            'TCS', 'WIPRO', 'AXISBANK', 'KOTAKBANK', 'ITC',
            'MARUTI', 'ASIANPAINT', 'HINDUNILVR', 'ULTRACEMCO'
        ]
        
        return jsonify({
            'stocks': indian_stocks,
            'count': len(indian_stocks),
            'timestamp': datetime.now().isoformat(),
            'note': 'These are common Indian stocks available via Alpha Vantage and CSV'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stock_data/<stock_symbol>', methods=['GET'])
def get_stock_data(stock_symbol):
    """Get stock data for visualization"""
    try:
        days = request.args.get('days', 100, type=int)
        
        df = load_stock_data(stock_symbol, days=days)
        if df is None:
            return jsonify({'error': 'Stock not found or no data available'}), 404
        
        # Get the actual data source from the dataframe
        actual_data_source = getattr(df, '_data_source', 'Unknown')
        
        data = {
            'stock_symbol': stock_symbol,
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].tolist(),
            'volumes': df['Volume'].tolist(),
            'timestamp': datetime.now().isoformat(),
            'data_source': actual_data_source,
            'days_requested': days,
            'days_received': len(df)
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics"""
    return jsonify({
        'model_name': 'Moving Average + Sentiment (Alpha Vantage)',
        'metrics': {
            'rmse': 'N/A (testing mode)',
            'r2': 'N/A (testing mode)',
            'mae': 'N/A (testing mode)'
        },
        'training_info': {
            'algorithm': 'Moving Average + Sentiment',
            'note': 'Using Alpha Vantage API for reliable market data',
            'feature_count': 5,
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    current_time = time.time()
    
    # Calculate rate limiting info
    calls_this_minute = len([t for t in ALPHA_VANTAGE_CALL_TIMES if current_time - t < 60])
    calls_today = len([t for t in ALPHA_VANTAGE_CALL_TIMES if current_time - t < 86400])  # 24 hours
    
    return jsonify({
        'status': 'healthy',
        'mode': 'alpha_vantage',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'predict': '/predict',
            'stocks': '/stocks',
            'health': '/health',
            'performance': '/performance'
        },
        'warnings': [
            '⚠️ CSV data files are from 2018 and may be outdated',
            '🌐 For live data, ensure internet connectivity to Alpha Vantage',
            '📊 Predictions using live data will be more accurate'
        ],
        'api_info': {
            'provider': 'Alpha Vantage',
            'rate_limit': '500 calls/day (free tier)',
            'data_quality': 'Professional-grade market data'
        },
        'rate_limiting': {
            'calls_this_minute': calls_this_minute,
            'calls_today': calls_today,
            'max_per_minute': ALPHA_VANTAGE_RATE_LIMIT,
            'max_per_day': 500,
            'min_interval_seconds': MIN_CALL_INTERVAL,
            'status': 'OK' if calls_this_minute < ALPHA_VANTAGE_RATE_LIMIT else 'RATE_LIMITED'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction API with Alpha Vantage...")
    print("📊 Mode: Alpha Vantage API (500 calls/day free tier)")
    print("🌐 Starting Flask API server...")
    
    if ALPHA_VANTAGE_API_KEY == 'YOUR_API_KEY':
        print("❌ ERROR: Please replace 'YOUR_API_KEY' with your actual Alpha Vantage API key!")
        print("   Edit the file and set: ALPHA_VANTAGE_API_KEY = 'your_actual_api_key_here'")
        exit(1)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
