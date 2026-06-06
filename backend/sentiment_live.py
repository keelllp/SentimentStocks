"""
Live per-symbol news sentiment with per-day on-disk caching.
Falls back to neutral sentiment on any failure so /predict never hangs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date
from typing import Optional

# Import SentimentAnalyzer from notebooks/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'notebooks'))

try:
    from sentiment_analysis import SentimentAnalyzer
    _analyzer: Optional[SentimentAnalyzer] = SentimentAnalyzer()
except Exception:
    _analyzer = None

try:
    from gnews import GNews
    _gnews_available = True
except ImportError:
    _gnews_available = False

_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'news_cache')
_NEUTRAL = {
    'Positive_mean': 0.0,
    'Negative_mean': 0.0,
    'Neutral_mean': 1.0,
    'news_count': 0,
    'news_count_7d_avg': 0.0,
    'news_count_30d_avg': 0.0,
}

_log = logging.getLogger('sentimentstocks')


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(symbol: str, day: date) -> str:
    return os.path.join(_CACHE_DIR, f'{symbol}_{day.isoformat()}.json')


def _read_cache(symbol: str, day: date) -> Optional[dict]:
    path = _cache_path(symbol, day)
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _write_cache(symbol: str, day: date, data: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        with open(_cache_path(symbol, day), 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        _log.warning(f'Sentiment cache write failed for {symbol}: {e}')


# ── Score a list of titles ─────────────────────────────────────────────────────

def _score_titles(titles: list[str]) -> dict:
    """Average VADER pos/neg/neu across all titles. Returns sentiment dict."""
    if not _analyzer or not titles:
        return dict(_NEUTRAL)

    pos_scores, neg_scores, neu_scores = [], [], []
    for title in titles:
        if not title:
            continue
        try:
            s = _analyzer.get_combined_sentiment(title)
            pos_scores.append(s.get('vader_positive', 0.0))
            neg_scores.append(s.get('vader_negative', 0.0))
            neu_scores.append(s.get('vader_neutral', 1.0))
        except Exception:
            continue

    if not pos_scores:
        return dict(_NEUTRAL)

    n = len(pos_scores)
    return {
        'Positive_mean': round(sum(pos_scores) / n, 4),
        'Negative_mean': round(sum(neg_scores) / n, 4),
        'Neutral_mean': round(sum(neu_scores) / n, 4),
        'news_count': n,
        'news_count_7d_avg': float(n),
        'news_count_30d_avg': float(n),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def get_live_sentiment(symbol: str) -> dict:
    """
    Fetch and score today's news headlines for `symbol`.
    Result is cached per-symbol per-day on disk.
    Always returns a valid dict (falls back to neutral on any error).
    """
    today = date.today()

    # Cache hit
    cached = _read_cache(symbol, today)
    if cached is not None:
        _log.info(f'Sentiment: cache hit for {symbol} ({today})')
        return cached

    if not _gnews_available or _analyzer is None:
        _log.warning(f'Sentiment: gnews or SentimentAnalyzer unavailable, using neutral for {symbol}')
        result = dict(_NEUTRAL)
        _write_cache(symbol, today, result)
        return result

    _log.info(f'Sentiment: fetching live news for {symbol}')
    try:
        gn = GNews(language='en', country='IN', max_results=20, period='7d')
        articles = gn.get_news(f'{symbol} stock India')

        if not articles:
            _log.info(f'Sentiment: no articles for {symbol}, using neutral')
            result = dict(_NEUTRAL)
            _write_cache(symbol, today, result)
            return result

        titles = [a.get('title', '') for a in articles if a.get('title')]
        result = _score_titles(titles)
        _log.info(f'Sentiment: {symbol} — {result["news_count"]} articles, '
                  f'pos={result["Positive_mean"]:.3f} neg={result["Negative_mean"]:.3f}')
        _write_cache(symbol, today, result)
        return result

    except Exception as e:
        _log.warning(f'Sentiment: live fetch failed for {symbol}: {e}, using neutral')
        result = dict(_NEUTRAL)
        _write_cache(symbol, today, result)
        return result
