"""
SentimentStocks Flask API
"""

from __future__ import annotations

import os
import sys
import random
from datetime import datetime, timedelta

import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Bootstrap path + env ───────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _BACKEND_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
    load_dotenv(os.path.join(_BACKEND_DIR, '.env'))
except Exception:
    pass

# ── Internal imports ───────────────────────────────────────────────────────────
from logging_config import setup_logging
from angel_session import AngelSessionManager
from sentiment_live import get_live_sentiment
from production_predictor import ProductionPredictor

log = setup_logging()

# ── Angel One token map (NSE instrument tokens) ────────────────────────────────
SYMBOL_TOKEN_MAP: dict[str, str] = {
    'RELIANCE': '2885', 'INFY': '1594', 'TCS': '11536',
    'HDFCBANK': '1333', 'ICICIBANK': '4963', 'SBIN': '3045',
    'WIPRO': '3787', 'HCLTECH': '7229', 'ITC': '1660',
    'BHARTIARTL': '2712', 'TATASTEEL': '3499',
}

# ── Singletons ─────────────────────────────────────────────────────────────────
angel_manager = AngelSessionManager()
predictor = ProductionPredictor()
log.info(f'Predictor ready: {predictor.model_name}')

# ── CSV fallback ───────────────────────────────────────────────────────────────
_csv_df: pd.DataFrame | None = None
_csv_symbols: list[str] = []
AVAILABLE_STOCKS: list[str] = []


def _csv_path() -> str:
    return os.path.normpath(os.path.join(_PROJECT_ROOT, 'processed_data', 'final_cleaned_data.csv'))


def _load_csv_once() -> None:
    global _csv_df, _csv_symbols, AVAILABLE_STOCKS
    if _csv_df is not None:
        return
    path = _csv_path()
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
        needed = {'Stock', 'Datetime', 'Open', 'High', 'Low', 'Close'}
        if not needed.issubset(set(df.columns)):
            return
        _csv_df = df
        syms = list(dict.fromkeys(df['Stock'].astype(str).tolist()))
        _csv_symbols = syms[:12]
        if not AVAILABLE_STOCKS:
            AVAILABLE_STOCKS[:] = list(_csv_symbols)
        log.info(f'CSV fallback loaded: {len(_csv_symbols)} stocks, {len(df):,} rows')
    except Exception as e:
        log.error(f'CSV load failed: {e}')


def _get_csv_symbols() -> list[str]:
    _load_csv_once()
    return list(_csv_symbols)


def _get_csv_series(symbol: str, days: int = 100) -> dict | None:
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
        points = [{'time': d.isoformat(), 'close': float(c)}
                  for d, c in zip(sdf['Date'], sdf['Close'])]
        data_points = [{'timestamp': p['time'], 'price': p['close']} for p in points]
        candles = [{
            't': pd.to_datetime(row['Date']).isoformat(),
            'o': float(row['Open']),
            'h': float(row['High']),
            'l': float(row['Low']),
            'c': float(row['Close']),
            'v': float(row.get('Volume', 0)),
        } for _, row in sdf.iterrows()]
        last_close = float(sdf['Close'].iloc[-1])
        return {
            'current_price': last_close,
            'series': points,
            'data': data_points,
            'candles': candles,
            'timestamp': pd.to_datetime(sdf['Date'].iloc[-1]).isoformat(),
        }
    except Exception as e:
        log.error(f'CSV series error for {symbol}: {e}')
        return None


# ── Angel One helpers ──────────────────────────────────────────────────────────

def get_ltp(symbol: str) -> float | None:
    for attempt in range(2):
        obj = angel_manager.get_client()
        if obj is None:
            return None
        token = SYMBOL_TOKEN_MAP.get(symbol.upper())
        if not token:
            log.warning(f'No token configured for {symbol}')
            return None
        try:
            resp = obj.ltpData(exchange='NSE',
                               tradingsymbol=f'{symbol.upper()}-EQ',
                               symboltoken=token)
            if isinstance(resp, dict) and resp.get('success') is False:
                msg = resp.get('message', '')
                if 'Invalid Token' in msg or 'Unauthorised' in msg:
                    if attempt == 0:
                        angel_manager.invalidate()
                        continue
                    return None
                if 'Failed to get symbol details' in msg:
                    resp = obj.ltpData(exchange='NSE',
                                       tradingsymbol=symbol.upper(),
                                       symboltoken=token)
            if isinstance(resp, dict):
                data = resp.get('data') or {}
                if isinstance(data, dict) and 'ltp' in data:
                    ltp = float(data['ltp'])
                    log.info(f'LTP {symbol}: ₹{ltp:,.2f}')
                    return ltp
            return None
        except Exception as e:
            log.error(f'LTP error for {symbol}: {e}')
            return None
    return None


def get_historical_ohlc(symbol: str, interval: str = 'ONE_DAY', days: int = 100):
    for attempt in range(2):
        obj = angel_manager.get_client()
        if obj is None:
            return None
        token = SYMBOL_TOKEN_MAP.get(symbol.upper())
        if not token:
            return None
        try:
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=days)
            params = {
                'exchange': 'NSE',
                'symboltoken': token,
                'interval': interval,
                'fromdate': start_date.strftime('%Y-%m-%d %H:%M'),
                'todate': end_date.strftime('%Y-%m-%d %H:%M'),
                'tradingsymbol': f'{symbol.upper()}-EQ',
            }
            log.info(f'Fetching {days}d history for {symbol}')
            resp = obj.getCandleData(params)
            if isinstance(resp, dict) and resp.get('status') is False:
                msg = resp.get('message', '')
                if 'Invalid Token' in msg or 'Unauthorised' in msg:
                    if attempt == 0:
                        angel_manager.invalidate()
                        continue
                    return None
                if 'Failed to get symbol details' in msg or 'Invalid' in msg:
                    params['tradingsymbol'] = symbol.upper()
                    resp = obj.getCandleData(params)
            if not isinstance(resp, dict) or resp.get('status') is False:
                return None
            candles = resp.get('data', [])
            log.info(f'History {symbol}: {len(candles)} candles')
            return candles if candles else None
        except Exception as e:
            log.error(f'History error for {symbol}: {e}')
            return None
    return None


# ── Utility ────────────────────────────────────────────────────────────────────

def _next_trading_date(n: int = 3) -> str:
    """Return the date string n trading days from now (skips weekends)."""
    d = datetime.now().date()
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d.strftime('%Y-%m-%d')


def _confidence_label(p_up: float) -> str:
    """Map a UP-probability to 'high' / 'medium' / 'low'."""
    certainty = abs(p_up - 0.5) * 2  # 0=random, 1=certain
    if certainty >= 0.30:
        return 'high'
    elif certainty >= 0.12:
        return 'medium'
    return 'low'


def _build_prediction_response(
    symbol: str,
    result: dict,
    data_source: str,
) -> dict:
    """Assemble the JSON response dict the frontend expects."""
    model_name = predictor.metadata.get('model_name', result.get('method', 'LightGBM Ensemble'))
    conf_float = result.get('confidence', 0.5)
    conf_label = _confidence_label(conf_float) if isinstance(conf_float, float) else 'medium'
    horizon = predictor.horizon if hasattr(predictor, 'horizon') else 3

    return {
        'stock_symbol': symbol,
        'current_price': result['current_price'],
        'predicted_price': result['predicted_price'],
        'current_price_formatted': f"{result['current_price']:,.2f}",
        'predicted_price_formatted': f"{result['predicted_price']:,.2f}",
        'price_change': result['price_change'],
        'price_change_pct': result['price_change_pct'],
        'prediction_date': _next_trading_date(horizon),
        'model_info': {
            'name': model_name,
            'method': result.get('method', ''),
            'features_used': result.get('features_used', 0),
            'rmse': 'N/A',
            'r2': 'N/A',
            'signal': result.get('signal', ''),
            'horizon': result.get('horizon', f'{horizon} Days'),
        },
        'confidence': conf_label,
        'confidence_score': round(conf_float, 4) if isinstance(conf_float, float) else None,
        'sentiment_score': None,
        'data_source': data_source,
        'timestamp': datetime.now().isoformat(),
    }


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-me-in-production')
_load_csv_once()


@app.route('/')
def root():
    meta = predictor.metadata
    return jsonify({
        'message': 'SentimentStocks API — AI-Powered Stock Predictions',
        'version': '3.0',
        'model': predictor.model_name,
        'endpoints': {
            'predict': '/predict',
            'stocks': '/stocks',
            'stock_data': '/stock_data/<symbol>',
            'health': '/health',
            'performance': '/performance',
        },
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': predictor.model_name,
        'angel_auth': angel_manager._client is not None,
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/performance')
def performance():
    """Return real model metrics from model_metadata.json."""
    if not predictor.metadata:
        return jsonify({'error': 'No model metadata available — run train_model.py first'}), 503
    return jsonify(predictor.metadata)


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """No-op endpoint kept for frontend compatibility."""
    return jsonify({'status': 'ok'})


@app.route('/stocks')
def stocks():
    csv_syms = _get_csv_symbols()
    symbols = csv_syms if csv_syms else list(SYMBOL_TOKEN_MAP.keys())
    return jsonify({'stocks': symbols, 'count': len(symbols), 'timestamp': datetime.now().isoformat()})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    symbol = (data.get('stock_symbol') or data.get('symbol') or 'INFY').upper()
    log.info(f'/predict {symbol}')

    # ── Tier 1: Angel One live data + LightGBM ─────────────────────────────────
    price = get_ltp(symbol)
    if price is not None:
        candles = get_historical_ohlc(symbol, interval='ONE_DAY', days=260)
        if candles and len(candles) >= 50:
            df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['Date'] = pd.to_datetime(df['time'])
            df = df.set_index('Date').sort_index()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.rename(columns={'open': 'Open', 'high': 'High',
                                    'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

            sentiment = get_live_sentiment(symbol)
            result = predictor.predict(symbol, df, sentiment_dict=sentiment)
            if result is not None:
                log.info(f'Prediction {symbol}: {result["signal"]} '
                         f'₹{result["current_price"]:,.2f} → ₹{result["predicted_price"]:,.2f} '
                         f'({result["price_change_pct"]:+.2f}%)')
                return jsonify(_build_prediction_response(
                    symbol, result, 'Angel One SmartAPI + LightGBM'
                ))

        # Live price but no sufficient history — return price-only response
        log.info(f'{symbol}: live LTP only (insufficient history)')
        return jsonify({
            'stock_symbol': symbol,
            'current_price': float(price),
            'predicted_price': float(price),
            'current_price_formatted': f'{float(price):,.2f}',
            'predicted_price_formatted': f'{float(price):,.2f}',
            'price_change': 0.0,
            'price_change_pct': 0.0,
            'prediction_date': _next_trading_date(predictor.horizon),
            'model_info': {'name': 'Live Price Only', 'rmse': 'N/A', 'r2': 'N/A'},
            'confidence': 'low',
            'data_source': 'Angel One SmartAPI',
            'timestamp': datetime.now().isoformat(),
        })

    # ── Tier 2: CSV fallback + LightGBM ───────────────────────────────────────
    fallback = _get_csv_series(symbol, days=260)
    if fallback is None:
        return jsonify({'error': f'No data available for {symbol}'}), 404

    df = pd.DataFrame(fallback['series'])
    df['Date'] = pd.to_datetime(df['time'])
    df = df.set_index('Date')
    df['Close'] = df['close']
    df['Open'] = df['close']
    df['High'] = df['close'] * 1.01
    df['Low'] = df['close'] * 0.99
    df['Volume'] = 1_000_000.0

    sentiment = get_live_sentiment(symbol)
    result = predictor.predict(symbol, df, sentiment_dict=sentiment)
    if result is not None:
        log.info(f'Prediction (CSV) {symbol}: {result["signal"]} '
                 f'₹{result["current_price"]:,.2f} → ₹{result["predicted_price"]:,.2f}')
        return jsonify(_build_prediction_response(
            symbol, result, 'CSV Data + LightGBM'
        ))

    # ── Tier 3: Simple drift ───────────────────────────────────────────────────
    current_price = float(fallback['current_price'])
    closes = [pt['close'] for pt in fallback['series']]
    if len(closes) >= 6:
        recent = closes[-6:-1]
        avg = sum(recent) / len(recent)
        drift = (current_price - avg) / avg if avg else 0.0
        predicted_price = round(current_price * (1 + 0.2 * drift), 2)
    else:
        predicted_price = current_price

    change = round(predicted_price - current_price, 2)
    change_pct = round((change / current_price) * 100, 2) if current_price else 0.0
    return jsonify({
        'stock_symbol': symbol,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'current_price_formatted': f'{current_price:,.2f}',
        'predicted_price_formatted': f'{predicted_price:,.2f}',
        'price_change': change,
        'price_change_pct': change_pct,
        'prediction_date': _next_trading_date(predictor.horizon),
        'model_info': {'name': 'Drift Fallback', 'rmse': 'N/A', 'r2': 'N/A'},
        'confidence': 'low',
        'data_source': 'CSV Data',
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/stock_data/<symbol>')
def stock_data(symbol: str):
    symbol = symbol.upper()
    log.info(f'/stock_data {symbol}')

    price = get_ltp(symbol)
    if price is None:
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
            'data_source': 'CSV Data (yfinance)',
        })

    # Live: synthetic intraday series around current LTP
    now = datetime.now()
    points = []
    for i in range(30, -1, -1):
        t = now - pd.Timedelta(minutes=i)
        jitter = (random.random() - 0.5) * 0.005 * float(price)
        close = round(float(price) + jitter, 2)
        points.append({'time': t.isoformat(), 'close': close})

    candles = [{
        't': p['time'],
        'o': p['close'], 'h': round(p['close'] * 1.001, 2),
        'l': round(p['close'] * 0.999, 2), 'c': p['close'], 'v': 0,
    } for p in points]

    return jsonify({
        'stock_symbol': symbol,
        'current_price': price,
        'series': points,
        'data': [{'timestamp': p['time'], 'price': p['close']} for p in points],
        'candles': candles,
        'dates': [p['time'] for p in points],
        'prices': [p['close'] for p in points],
        'volumes': [0] * len(points),
        'timestamp': now.isoformat(),
        'data_source': 'Angel One SmartAPI',
    })


if __name__ == '__main__':
    log.info('Starting SentimentStocks API on http://0.0.0.0:5000')
    log.info(f'Model: {predictor.model_name}')
    log.info('Tip: tail backend/logs/app.log to follow live logs')
    app.run(debug=False, host='0.0.0.0', port=5000)
