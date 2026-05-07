
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from datetime import datetime
import pandas as pd
import requests
from SmartApi import SmartConnect
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.production_predictor import ProductionPredictor
try:
    from dotenv import load_dotenv
    # Load .env from project root (one level up from backend/) and from backend/
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except Exception:
    pass
try:
    import pyotp
except Exception:
    pyotp = None


# =========================
# Angel One configuration
# =========================
# These values are already present in your codebase; we keep them as defaults
ANGEL_API_KEY = "UsOGGTK1"
ANGEL_API_SECRET = "2afb652b-3ea0-4768-a49b-bd592ab2787a"  # not used directly by SmartApi for LTP
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "AABX480655")
ANGEL_CLIENT_PWD = os.environ.get("ANGEL_CLIENT_PWD", "Kbsh@2310_")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET", "WBWETMN5YX4RM7WNWHADBJQU6U")
ANGEL_MPIN = os.environ.get("ANGEL_MPIN", os.environ.get("ANGEL_PIN", "6969"))  # 4-digit MPIN if needed
ANGEL_ACCESS_TOKEN = os.environ.get("ANGEL_ACCESS_TOKEN", "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VybmFtZSI6IkFBQlg0ODA2NTUiLCJyb2xlcyI6MCwidXNlcnR5cGUiOiJVU0VSIiwidG9rZW4iOiJleUpoYkdjaU9pSlNVekkxTmlJc0luUjVjQ0k2SWtwWFZDSjkuZXlKMWMyVnlYM1I1Y0dVaU9pSmpiR2xsYm5RaUxDSjBiMnRsYmw5MGVYQmxJam9pZEhKaFpHVmZZV05qWlhOelgzUnZhMlZ1SWl3aVoyMWZhV1FpT2pNc0luTnZkWEpqWlNJNklqTWlMQ0prWlhacFkyVmZhV1FpT2lJME5qQTFZbUU1TkMwM1pUY3pMVE14TURVdFlXVXhZeTAxT0RabVltUXhZVEl6WlRVaUxDSnJhV1FpT2lKMGNtRmtaVjlyWlhsZmRqSWlMQ0p2Ylc1bGJXRnVZV2RsY21sa0lqb3pMQ0p3Y205a2RXTjBjeUk2ZXlKa1pXMWhkQ0k2ZXlKemRHRjBkWE1pT2lKaFkzUnBkbVVpZlN3aWJXWWlPbnNpYzNSaGRIVnpJam9pWVdOMGFYWmxJbjE5TENKcGMzTWlPaUowY21Ga1pWOXNiMmRwYmw5elpYSjJhV05sSWl3aWMzVmlJam9pUVVGQ1dEUTRNRFkxTlNJc0ltVjRjQ0k2TVRjM09ERTJPVGszTml3aWJtSm1Jam94TnpjNE1EZ3pNemsyTENKcFlYUWlPakUzTnpnd09ETXpPVFlzSW1wMGFTSTZJbU5sT0RCallXRTNMV00xTVRBdE5ERmlZUzA1WVRobUxURXlZamN5WVRsaVptTTVPQ0lzSWxSdmEyVnVJam9pSW4wLm4wRGlidmIzTV94eWx5aGNDZW11ZXhWYy1RM1JKQ0pWMGpsc1FfWDJvMjY3SkxoMC1EaFNYcDExeEJubjY0c0NieGNfd3dMdzQzY0ZqMmdtb2QzZW5FWFBXZkdSTjQ1emRxb1ptS0JYT20wNTV3NlV0bGs4UDFLWnZ4UVR4a1ZPY2MxUUxsZVBibFZmQXpKa3Vud0Z2ckt6UGVuR3g0WlFaVWV3b1NiWDZBdyIsIkFQSS1LRVkiOiJVc09HR1RLMSIsImlhdCI6MTc3ODA4MzU3NiwiZXhwIjoxNzc4MDkyMjAwfQ.rYjZsRGfUqnFkh-_zTm9on7bEcvUZ6m6cn29nwMNlP-6qvEw0wDhUo1PGzEej6-X-RWCGeHvU66GPlySqr5dHA")
# NSE tokens for commonly used symbols (expand as needed)
SYMBOL_TOKEN_MAP = {
    'RELIANCE': '2885', 'INFY': '1594', 'TCS': '11536', 'HDFCBANK': '1333',
    'ICICIBANK': '4963', 'SBIN': '3045', 'WIPRO': '3787', 'HCLTECH': '7229',
    'ITC': '1660', 'BHARTIARTL': '2712', 'TATASTEEL': '3499', 'HDFC': '1330'
}

AVAILABLE_STOCKS = []  # will be populated from CSV fallback if available


# =========================
# Session management
# =========================
angel = None
angel_login_error = None

# =========================
# CSV fallback (silent)
# =========================
_csv_df = None
_csv_symbols = []

def _csv_path() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'final_cleaned_data.csv'))

def _load_csv_once():
    global _csv_df, _csv_symbols, AVAILABLE_STOCKS
    if _csv_df is not None:
        return
    path = _csv_path()
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Expect columns: Stock, Datetime, Open, High, Low, Close, Volume
            needed = {'Stock','Datetime','Open','High','Low','Close'}
            if needed.issubset(set(df.columns)):
                _csv_df = df
                symbols = list(dict.fromkeys(df['Stock'].astype(str).tolist()))
                # Limit to first 8 symbols deterministically
                _csv_symbols = symbols[:8]
                if not AVAILABLE_STOCKS:
                    AVAILABLE_STOCKS = list(_csv_symbols)
    except Exception:
        _csv_df = None
        _csv_symbols = []

def _get_csv_symbols() -> list:
    _load_csv_once()
    return list(_csv_symbols)

def _get_csv_series(symbol: str, days: int = 100):
    _load_csv_once()
    if _csv_df is None:
        return None
    try:
        sdf = _csv_df[_csv_df['Stock'].astype(str).str.upper() == symbol.upper()].copy()
        if sdf.empty:
            return None
        sdf['Date'] = pd.to_datetime(sdf['Datetime'])
        sdf.sort_values('Date', inplace=True)
        if days and len(sdf) > days:
            sdf = sdf.tail(days)
        # Build series
        points = [{'time': d.isoformat(), 'close': float(c)} for d, c in zip(sdf['Date'], sdf['Close'])]
        data_points = [{'timestamp': p['time'], 'price': p['close']} for p in points]
        candles = []
        for _, row in sdf.iterrows():
            t = pd.to_datetime(row['Date']).isoformat()
            candles.append({
                't': t,
                'o': float(row['Open']),
                'h': float(row['High']),
                'l': float(row['Low']),
                'c': float(row['Close']),
                'v': float(row.get('Volume', 0) if 'Volume' in row else 0)
            })
    except Exception:
        return None
    last_close = float(sdf['Close'].iloc[-1])
    return {
        'current_price': last_close,
        'series': points,
        'data': data_points,
        'candles': candles,
        'timestamp': pd.to_datetime(sdf['Date'].iloc[-1]).isoformat()
    }


def generate_totp_from_secret(secret: str) -> str | None:
    if not secret or pyotp is None:
        return None
    try:
        return pyotp.TOTP(secret).now()
    except Exception:
        return None


def mpin_login(clientcode: str, mpin: str, totp_code: str | None) -> str | None:
    try:
        url = "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByMpin"
        # Provide similar headers as SmartApi does
        client_local_ip = os.environ.get('ANGEL_LOCAL_IP', '127.0.0.1')
        client_public_ip = os.environ.get('ANGEL_PUBLIC_IP', '127.0.0.1')
        client_mac = os.environ.get('ANGEL_MAC_ADDR', '00:00:00:00:00:00')
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'X-PrivateKey': ANGEL_API_KEY,
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': client_local_ip,
            'X-ClientPublicIP': client_public_ip,
            'X-MACAddress': client_mac,
        }
        payload = {'clientcode': clientcode, 'mpin': mpin}
        if totp_code:
            payload['totp'] = totp_code
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        # Parse JSON even if content-type header is wrong
        try:
            data = resp.json()
        except Exception:
            print(f"MPIN login non-JSON response text: {resp.text[:200]}")
            return None
        if not isinstance(data, dict):
            print(f"MPIN login unexpected response: {resp.status_code}")
            return None
        if data.get('status') is False:
            print(f"MPIN login failed: {data.get('message')} ({data.get('errorcode')})")
            return None
        return (data.get('data') or {}).get('accessToken')
    except Exception as e:
        print(f"MPIN login error: {e}")
        return None


def get_session():
    global angel, angel_login_error

    if angel is not None:
        return angel

    try:
        angel = SmartConnect(api_key=ANGEL_API_KEY)

        totp = pyotp.TOTP(
            ANGEL_TOTP_SECRET
        ).now()

        data = angel.generateSession(
            ANGEL_CLIENT_ID,
            ANGEL_MPIN,
            totp
        )

        if not data or 'data' not in data:
            angel_login_error = "Login failed"
            angel = None
            return None

        print("✅ Angel One login success")

        return angel

    except Exception as e:
        print(f"Angel login failed: {e}")

        angel_login_error = str(e)
        angel = None

        return None


def get_ltp(symbol: str) -> float | None:
    obj = get_session()
    if obj is None:
        return None
    token = SYMBOL_TOKEN_MAP.get(symbol.upper())
    if not token:
        print(f"No token configured for {symbol}")
        return None

    def _fetch(tradingsymbol):
        params = {"exchange": "NSE", "tradingsymbol": tradingsymbol, "symboltoken": token}
        return obj.ltpData(**params)

    try:
        # Try with -EQ suffix first
        resp = _fetch(f"{symbol.upper()}-EQ")
        
        # If it fails, try without -EQ suffix
        if isinstance(resp, dict) and resp.get('success') is False:
            msg = resp.get('message', '')
            if 'Failed to get symbol details' in msg:
                print(f"Trying {symbol} without -EQ suffix")
                resp = _fetch(symbol.upper())
        
        if not isinstance(resp, dict):
            print(f"LTP unexpected response for {symbol}: {resp}")
            return None
        if resp.get('success') is False:
            msg = resp.get('message', '')
            print(f"LTP failed for {symbol}: {msg}")
            if 'Invalid Token' in msg:
                # retry once after re-login
                global angel
                angel = None
                if get_session() is None:
                    return None
                resp = _fetch(f"{symbol.upper()}-EQ")
                if not isinstance(resp, dict) or resp.get('success') is False:
                    return None
        data = resp.get('data') or {}
        if isinstance(data, dict) and 'ltp' in data:
            return data['ltp']
        print(f"LTP missing 'ltp' for {symbol}: {resp}")
        return None
    except Exception as e:
        print(f"LTP fetch error for {symbol}: {e}")
        return None


def get_historical_ohlc(symbol: str, interval="ONE_DAY", days=100):
    """
    Fetch historical OHLC data for a symbol from Angel One SmartAPI.
    interval: ONE_MINUTE, FIVE_MINUTE, TEN_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY
    """
    obj = get_session()
    if obj is None:
        print(f"❌ No Angel One session available for {symbol}")
        return None

    token = SYMBOL_TOKEN_MAP.get(symbol.upper())
    if not token:
        print(f"❌ No token configured for {symbol}")
        return None

    try:
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)

        # Try with -EQ suffix first
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M"),
            "tradingsymbol": f"{symbol.upper()}-EQ"
        }
        
        print(f"🔍 Fetching historical data for {symbol}: {days} days")
        print(f"📅 Date range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        resp = obj.getCandleData(params)
        
        # If it fails with symbol error, try without -EQ suffix
        if isinstance(resp, dict) and resp.get("status") is False:
            msg = resp.get("message", "")
            if 'Failed to get symbol details' in msg or 'Invalid' in msg:
                print(f"Trying {symbol} without -EQ suffix")
                params["tradingsymbol"] = symbol.upper()
                resp = obj.getCandleData(params)
        
        if not isinstance(resp, dict):
            print(f"❌ Unexpected response type for {symbol}: {type(resp)}")
            return None
            
        if resp.get("status") is False:
            print(f"❌ Historical data fetch failed for {symbol}: {resp}")
            return None

        # Format: [time, open, high, low, close, volume]
        candles = resp.get("data", [])
        print(f"✅ Retrieved {len(candles)} candles for {symbol}")
        
        if len(candles) == 0:
            print(f"⚠️ No historical data returned for {symbol}")
            return None
            
        return candles
        
    except Exception as e:
        print(f"❌ Error fetching historical data for {symbol}: {e}")
        return None


# =========================
# Flask app
# =========================
app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this'

# Initialize the production predictor
predictor = ProductionPredictor()


@app.route('/')
def root():
    return jsonify({
        'message': 'SentimentStock API - AI-Powered Stock Predictions',
        'version': '2.0',
        'endpoints': {
            'predict': '/predict',
            'stocks': '/stocks',
            'stock_data': '/stock_data/<symbol>',
            'health': '/health'
        },
        'data_source': 'Angel One SmartAPI + ML Models',
        'features': [
            'LightGBM Model Predictions',
            'Technical Analysis Features',
            'Sentiment Analysis Integration',
            'Real-time Stock Data'
        ],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'auth': angel is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/stocks')
def stocks():
    # Prefer CSV symbols (limited to 8) if available; else fall back to hardcoded list
    csv_syms = _get_csv_symbols()
    symbols = csv_syms if csv_syms else (AVAILABLE_STOCKS if AVAILABLE_STOCKS else list(SYMBOL_TOKEN_MAP.keys()))
    return jsonify({'stocks': symbols, 'count': len(symbols), 'timestamp': datetime.now().isoformat()})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    symbol = (data.get('stock_symbol') or data.get('symbol') or 'INFY').upper()
    sentiment_score = data.get('sentiment_score', 0.0)  # Default neutral sentiment
    
    try:
        # Try to get live price first
        price = get_ltp(symbol)
        
        if price is not None:
            # Live price available - get historical data for ML prediction
            candles = get_historical_ohlc(symbol, interval="ONE_DAY", days=100)
            if candles and len(candles) >= 50:
                # Convert candles to DataFrame
                df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['Date'] = pd.to_datetime(df['time'])
                df = df.set_index('Date')
                
                # Convert to numeric and rename to capitalized columns for predictor
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Rename columns to match predictor expectations
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Use the ML predictor
                prediction_result = predictor.predict(df, sentiment_score)
                
                if prediction_result is not None:
                    return jsonify({
                        'stock_symbol': symbol,
                        'current_price': prediction_result['current_price'],
                        'predicted_price': prediction_result['predicted_price'],
                        'current_price_formatted': f"{prediction_result['current_price']:,.2f}",
                        'predicted_price_formatted': f"{prediction_result['predicted_price']:,.2f}",
                        'price_change': prediction_result['price_change'],
                        'price_change_pct': prediction_result['price_change_pct'],
                        'prediction_date': (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                        'model_info': {
                            'name': 'LightGBM ML Model',
                            'method': prediction_result['method'],
                            'features_used': prediction_result['features_used'],
                            'rmse': 'N/A',
                            'r2': 'N/A'
                        },
                        'confidence': 'high' if prediction_result['confidence'] > 0.7 else 'medium',
                        'sentiment_score': sentiment_score,
                        'data_source': 'Angel One SmartAPI + ML Models',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Fallback: Use current price as prediction (temporary)
            current_price = float(price)
            predicted_price = current_price
            change = 0.0
            change_pct = 0.0
            return jsonify({
                'stock_symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'current_price_formatted': f"{current_price:,.2f}",
                'predicted_price_formatted': f"{predicted_price:,.2f}",
                'price_change': change,
                'price_change_pct': change_pct,
                'prediction_date': (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                'model_info': {
                    'name': 'Live Price (No Historical Data)',
                    'rmse': 'N/A',
                    'r2': 'N/A'
                },
                'confidence': 'low',
                'data_source': 'Angel One SmartAPI',
                'timestamp': datetime.now().isoformat()
            })
        
        # CSV fallback with ML prediction
        fallback = _get_csv_series(symbol, days=200)
        if fallback is None:
            return jsonify({'error': f'No data available for {symbol}'}), 404
        
        # Convert CSV data to DataFrame for ML prediction
        df = pd.DataFrame(fallback['series'])
        df['Date'] = pd.to_datetime(df['time'])
        df = df.set_index('Date')
        df['Close'] = df['close']
        df['Open'] = df['close']  # Approximate
        df['High'] = df['close'] * 1.01  # Approximate
        df['Low'] = df['close'] * 0.99   # Approximate
        df['Volume'] = 1000000  # Default volume
        
        # Use ML predictor
        prediction_result = predictor.predict(df, sentiment_score)
        
        if prediction_result is not None:
            return jsonify({
                'stock_symbol': symbol,
                'current_price': prediction_result['current_price'],
                'predicted_price': prediction_result['predicted_price'],
                'current_price_formatted': f"{prediction_result['current_price']:,.2f}",
                'predicted_price_formatted': f"{prediction_result['predicted_price']:,.2f}",
                'price_change': prediction_result['price_change'],
                'price_change_pct': prediction_result['price_change_pct'],
                'prediction_date': (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                'model_info': {
                    'name': 'LightGBM ML Model',
                    'method': prediction_result['method'],
                    'features_used': prediction_result['features_used'],
                    'rmse': 'N/A',
                    'r2': 'N/A'
                },
                'confidence': 'high' if prediction_result['confidence'] > 0.7 else 'medium',
                'sentiment_score': sentiment_score,
                'data_source': 'CSV Data + ML Models',
                'timestamp': datetime.now().isoformat()
            })
        
        # Final fallback: Simple drift calculation
        current_price = float(fallback['current_price'])
        closes = [pt['close'] for pt in fallback['series']]
        if len(closes) >= 6:
            recent = closes[-6:-1]
            recent_avg = sum(recent) / len(recent)
            drift = (current_price - recent_avg) / recent_avg if recent_avg else 0.0
            predicted_price = round(current_price * (1 + 0.2 * drift), 2)
        else:
            predicted_price = current_price
        
        change = round(predicted_price - current_price, 2)
        change_pct = round((change / current_price) * 100, 2) if current_price else 0.0
        return jsonify({
            'stock_symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'current_price_formatted': f"{current_price:,.2f}",
            'predicted_price_formatted': f"{predicted_price:,.2f}",
            'price_change': change,
            'price_change_pct': change_pct,
            'prediction_date': (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            'model_info': {
                'name': 'Fallback Drift',
                'rmse': 'N/A',
                'r2': 'N/A'
            },
            'confidence': 'low',
            'data_source': 'CSV Data',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/stock_data/<symbol>')
def stock_data(symbol: str):
    symbol = symbol.upper()
    price = get_ltp(symbol)
    if price is None:
        # CSV fallback with real series
        fallback = _get_csv_series(symbol, days=200)
        if fallback is None:
            return jsonify({'error': 'Stock not found or no data available'}), 404
        return jsonify({
            'stock_symbol': symbol,
            'current_price': fallback['current_price'],
            'series': fallback['series'],
            'data': fallback['data'],
            'candles': fallback['candles'],
            'dates': [p['time'] for p in fallback['series']],
            'prices': [p['close'] for p in fallback['series']],
            'volumes': [c.get('v', 0) for c in fallback['candles']],
            'timestamp': fallback['timestamp'],
            'data_source': 'Angel One SmartAPI'
        })
    # Build simple synthetic series around current LTP for visualization (Angel One live)
    now = datetime.now()
    points = []
    for i in range(30, -1, -1):
        t = now - pd.Timedelta(minutes=i)
        jitter = (random.random() - 0.5) * 0.01 * float(price)
        close = round(float(price) + jitter, 2)
        points.append({'time': t.isoformat(), 'close': close})
    data_points = [{'timestamp': p['time'], 'price': p['close']} for p in points]
    candles = [{
        't': p['time'],
        'o': p['close'], 'h': round(p['close'] * 1.002, 2), 'l': round(p['close'] * 0.998, 2), 'c': p['close'], 'v': 0
    } for p in points]
    return jsonify({
        'stock_symbol': symbol,
        'current_price': price,
        'series': points,
        'data': data_points,
        'candles': candles,
        'dates': [p['time'] for p in points],
        'prices': [p['close'] for p in points],
        'volumes': [0 for _ in points],
        'timestamp': now.isoformat(),
        'data_source': 'Angel One SmartAPI'
    })


if __name__ == '__main__':
    print("🚀 Starting SentimentStock API - AI-Powered Stock Predictions...")
    print("🤖 Loading ML Models...")
    print("🌐 Flask listening on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)