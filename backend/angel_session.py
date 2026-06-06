"""
Seamless Angel One session manager.

Hierarchy on get_client():
  1. In-memory cached client (fastest)
  2. Load token from .angel_session.json + validate via getProfile()
  3. renewAccessToken() using cached refreshToken (no TOTP needed)
  4. Full TOTP generateSession() as last resort (once per expiry cycle)

Result: at most one TOTP login per token-lifetime, zero manual steps.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import pyotp
from SmartApi import SmartConnect
from dotenv import load_dotenv

# Load creds from both root .env and backend/.env
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'), override=False)
load_dotenv(os.path.join(_BACKEND_DIR, '.env'), override=False)

API_KEY = os.getenv('API_KEY', '')
CLIENT_CODE = os.getenv('CLIENT_CODE', '')
MPIN = os.getenv('MPIN', '')
TOTP_SECRET = os.getenv('TOTP_SECRET', '')

_CACHE_FILE = os.path.join(_BACKEND_DIR, '.angel_session.json')


class AngelSessionManager:
    """Thread-safe, self-healing Angel One session."""

    def __init__(self):
        self._client: Optional[SmartConnect] = None
        self._lock = threading.Lock()

    # ── internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _log() -> logging.Logger:
        return logging.getLogger('sentimentstocks')

    def _load_cache(self) -> Optional[dict]:
        if not os.path.exists(_CACHE_FILE):
            return None
        try:
            with open(_CACHE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, jwt: str, refresh: str, feed: str) -> None:
        try:
            with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'jwtToken': jwt,
                    'refreshToken': refresh,
                    'feedToken': feed,
                    'client_code': CLIENT_CODE,
                }, f)
        except Exception as e:
            self._log().warning(f'Angel: could not write session cache: {e}')

    def _clear_cache(self) -> None:
        self._client = None
        try:
            if os.path.exists(_CACHE_FILE):
                os.remove(_CACHE_FILE)
        except Exception:
            pass

    def _make_client(self, jwt: str, refresh: str, feed: str) -> SmartConnect:
        obj = SmartConnect(API_KEY)
        obj.setAccessToken(jwt)
        obj.setRefreshToken(refresh)
        obj.setFeedToken(feed)
        return obj

    def _validate(self, obj: SmartConnect, refresh: str) -> bool:
        """Return True if the token on `obj` is still accepted by Angel One."""
        try:
            resp = obj.getProfile(refresh)
            if isinstance(resp, dict) and resp.get('status') is not False:
                return True
            return False
        except Exception:
            return False

    def _renew(self, obj: SmartConnect) -> bool:
        """Refresh jwt using the stored refreshToken. Returns True on success."""
        log = self._log()
        try:
            result = obj.renewAccessToken()
            if not result or 'jwtToken' not in result:
                log.warning('Angel: renewAccessToken returned no jwtToken')
                return False
            new_jwt = result['jwtToken']
            new_refresh = result.get('refreshToken', '')
            obj.setAccessToken(new_jwt)
            if new_refresh:
                obj.setRefreshToken(new_refresh)
            feed = getattr(obj, 'feed_token', '') or ''
            self._save_cache(new_jwt, new_refresh or '', feed)
            log.info('Angel: token renewed (no TOTP needed)')
            return True
        except Exception as e:
            log.warning(f'Angel: renewal failed: {e}')
            return False

    def _totp_login(self) -> Optional[SmartConnect]:
        """Full TOTP login — last resort."""
        log = self._log()
        if not all([API_KEY, CLIENT_CODE, MPIN, TOTP_SECRET]):
            log.error('Angel: credentials incomplete (API_KEY/CLIENT_CODE/MPIN/TOTP_SECRET)')
            return None
        try:
            totp = pyotp.TOTP(TOTP_SECRET).now()
            obj = SmartConnect(API_KEY)
            data = obj.generateSession(CLIENT_CODE, MPIN, totp)
            if not isinstance(data, dict) or data.get('status') is False or 'data' not in data:
                log.error(f'Angel: TOTP login failed: {data}')
                return None
            d = data['data']
            jwt = d.get('jwtToken', '')
            refresh = d.get('refreshToken', '')
            feed = d.get('feedToken', '')
            obj.setFeedToken(feed)
            obj.setRefreshToken(refresh)
            self._save_cache(jwt, refresh, feed)
            log.info('Angel: TOTP login successful — session cached')
            return obj
        except Exception as e:
            log.error(f'Angel: TOTP login error: {e}')
            return None

    # ── public API ─────────────────────────────────────────────────────────────

    def get_client(self) -> Optional[SmartConnect]:
        """
        Return a valid SmartConnect client, or None if all auth paths fail.
        Thread-safe; at most one TOTP call per token-lifetime.
        """
        with self._lock:
            # 1. In-memory client — assume still valid until a call fails
            if self._client is not None:
                return self._client

            # 2. Restore from cache + validate
            cache = self._load_cache()
            if cache and cache.get('jwtToken'):
                obj = self._make_client(
                    cache['jwtToken'],
                    cache.get('refreshToken', ''),
                    cache.get('feedToken', ''),
                )
                if self._validate(obj, cache.get('refreshToken', '')):
                    self._log().info('Angel: reusing cached session (no network login)')
                    self._client = obj
                    return obj

                # 3. Cache exists but token expired — try renewal
                self._log().info('Angel: cached token expired, attempting renewal')
                if self._renew(obj):
                    self._client = obj
                    return obj

            # 4. Fall through to full TOTP login
            self._log().info('Angel: performing full TOTP login')
            self._clear_cache()
            obj = self._totp_login()
            if obj:
                self._client = obj
            return obj

    def invalidate(self) -> None:
        """
        Call after receiving 'Invalid Token' from any API call so the next
        get_client() attempt starts fresh (renewal → TOTP as needed).
        """
        with self._lock:
            self._log().info('Angel: session invalidated — will re-authenticate on next call')
            self._clear_cache()
