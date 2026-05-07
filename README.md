# SentimentStocks

SentimentStocks is a stock price prediction system that combines market OHLC data with social media sentiment to predict next-day prices. This README contains setup, development, deployment, and troubleshooting instructions.

Contents
--------
- `backend/` — Flask API and production predictor (`main.py`, `production_predictor.py`).
- `frontend/` — React + TypeScript SPA (source in `src/`, production bundle in `build/`).
- `models/` — Trained artifacts: `optimized_lgb_model.joblib`, `enhanced_robust_scaler.joblib`, `feature_info.joblib`.
- `data/` — Raw and processed CSVs used by notebooks and for fallback.
- `notebooks/` — Jupyter notebooks for preprocessing and training.
- `src/` — Utility modules; `src/sentiment_analysis.py` is used by notebooks.

Quick links
-----------
- Backend entry: `backend/main.py`
- Predictor wrapper: `backend/production_predictor.py`
- Frontend entry (dev): `frontend/src/index.tsx`
- Model artifacts: `models/`
- Notebooks: `notebooks/`

Overview
--------
- The backend fetches historical OHLC data (via Angel One SmartAPI), computes technical features and aggregates Twitter sentiment, and serves next-day price predictions from a trained LightGBM model.
- The frontend lets users select a stock, request a prediction, and view charts + model information.

Detailed setup — Backend (development)
--------------------------------------
1. Create and activate a Python virtual environment (Windows cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install backend dependencies:

```cmd
cd backend
pip install -r requirements.txt
```

3. Configure environment variables. The backend reads from environment or `.env` files. Create a `.env` at the project root with values like:

```env
ANGEL_CLIENT_ID=AABX480655
ANGEL_CLIENT_PWD=your_password_here
ANGEL_TOTP_SECRET=YOUR_TOTP_SECRET  # optional
ANGEL_MPIN=8080                     # optional
ANGEL_ACCESS_TOKEN=eyJ...           # optional pre-generated token
```

4. Start the backend service:

```cmd
cd backend
python main.py
```

The backend logs will show whether the model and scaler loaded correctly.

Endpoints (summary)
-------------------
- `GET /` — API information
- `GET /health` — Health status (backend + model)
- `GET /stocks` — Available stocks
- `GET /stock_data/<symbol>` — Historical OHLC for charting (JSON)
- `POST /predict` — Generate prediction (JSON body). Example request:

```json
{ "stock_symbol": "INFY", "sentiment_score": 0.0 }
```

Production predictor notes
-------------------------
- `backend/production_predictor.py` loads `models/optimized_lgb_model.joblib`, `models/enhanced_robust_scaler.joblib`, and `models/feature_info.joblib`.
- It constructs technical indicators (MA, RSI, MACD, Bollinger Bands, volatility, lags) and uses the scaler and model to produce predictions.
- The predictor expects ~50–60 days of historical data to compute lag and moving-average features. If insufficient history is available, predictions may fall back to simpler heuristics.

Frontend (development & production)
-----------------------------------
Development:

```cmd
cd frontend
npm install
npm start
```

Open `http://localhost:3000`.

Production build:

```cmd
cd frontend
npm run build
```

The static files are generated in `frontend/build/` (deploy these to a static host).

Notes: I updated `frontend/public/index.html` title to `SentimentStocks`. Rebuild to regenerate production assets.

Data preprocessing & sentiment
------------------------------
- `notebooks/01_data_preprocessing.ipynb` reads raw tweets and computes sentiment using `src/sentiment_analysis.py`.
- `src/sentiment_analysis.py` uses VADER (with a small financial lexicon extension) and TextBlob. The notebook uses `SentimentAnalyzer.get_combined_sentiment(tweet)` and stores `vader_positive`, `vader_neutral`, `vader_negative` as `Positive`, `Neutral`, `Negative` columns.
- The notebook merges Twitter and stock data and forward-fills missing Twitter metrics with `df.fillna(method='ffill')`. For multi-stock dataframes, prefer group-wise forward fill to avoid data leakage:

```python
merged = merged.groupby('Stock').fillna(method='ffill')
```

Feature engineering & training
------------------------------
- `notebooks/02_feature_engineering_and_training.ipynb` shows the end-to-end training: feature computation, model training, CV, and evaluation.
- Many production-ready features are computed in `backend/production_predictor.py` for inference parity.

Testing and validation
----------------------
- Use the notebooks to reproduce preprocessing and training steps.
- To test the running API endpoints:

```cmd
curl http://localhost:5000/health

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{ \"stock_symbol\": \"INFY\" }"
```

Troubleshooting
---------------
- Model load errors: ensure `models/` contains the `.joblib` artifacts and the backend has read permission.
- Trivial predictions: verify at least 50–60 days of OHLC data are available for the requested symbol.
- Angel One authentication: check `.env` values, verify access token or MPIN/TOTP as appropriate, and review backend logs.

Developer notes
---------------
- `src/` contains utility modules. Only `src/sentiment_analysis.py` is used by the notebooks; other files (`data_processing.py`, `feature_engineering.py`, `models.py`, `evaluation.py`) are present but not wired into the backend. Consider refactoring notebooks to import these helpers for better reuse.
- To add an NSE token mapping, update `SYMBOL_TOKEN_MAP` in `backend/main.py`.
- When updating the model, replace the files in `models/` and restart the backend.

Recommended next steps
----------------------
1. Add automated unit tests for `production_predictor.prepare_features()` and `get_prediction()`.
2. Add a `Dockerfile` + `docker-compose.yml` for reproducible local development (I can add these if you want).
3. Centralize secrets (avoid committing tokens in code) and use CI secrets for deployment.

Contributing
------------
1. Fork the repository
2. Create a feature branch
3. Run tests (add a test suite) and formatters
4. Submit a pull request with a clear description and tests

License
-------
MIT

If you'd like, I can now:
- regenerate the frontend build (`npm run build`) so the production bundle text matches the new project name, or
- add a Dockerfile and `docker-compose.yml` to run backend + frontend locally.
Tell me which and I'll implement it.
- Fetches historical OHLC data and live LTP (via Angel One SmartAPI) for supported stocks.
- Computes technical indicators and aggregates daily Twitter sentiment.
- Uses a trained LightGBM model to predict next-day price and returns prediction metadata.

**Quick start — backend (recommended)**
1. Create and activate a Python virtual environment (Windows cmd):
   ```cmd
   python -m venv .venv
   .venv\\Scripts\\activate
   ```
2. Install Python dependencies:
   ```cmd
   pip install -r backend/requirements.txt
   ```
3. Configure Angel One credentials (recommended) in environment variables or a `.env` file at the project root. Required env vars the backend checks for:
   - `ANGEL_CLIENT_ID` (e.g. `AABX480655`)
   - `ANGEL_CLIENT_PWD` (your Angel One password)
   - `ANGEL_TOTP_SECRET` (optional, for TOTP)
   - `ANGEL_MPIN` (optional)
   - `ANGEL_ACCESS_TOKEN` (optional, pre-generated token)
4. Run the backend API:
   ```cmd
   cd backend
   python main.py
   ```

The backend exposes the following helpful endpoints:
- `GET /` — basic info
- `GET /health` — health check
- `GET /stocks` — list configured stocks
- `GET /stock_data/<symbol>` — historical OHLC for charting
- `POST /predict` — run prediction (body example: `{ \"stock_symbol\": \"INFY\" , \"sentiment_score\": 0.0 }`)

**Quick start — frontend (development)**
1. Install Node dependencies:
   ```cmd
   cd frontend
   npm install
   ```
2. Start dev server:
   ```cmd
   npm start
   ```
3. Open http://localhost:3000

Note: The `frontend/public/index.html` title has been updated to `SentimentStocks`. If you serve from `frontend/build/`, rebuild with `npm run build` to regenerate the production bundle.

**Models and features**
- Trained model artifacts live in `models/` and are loaded by `ProductionPredictor`.
- Feature engineering happens in `production_predictor.py` for production predictions; notebooks implement the training pipeline and include richer experiments.

**Data preprocessing & sentiment**
- `notebooks/01_data_preprocessing.ipynb` applies a `SentimentAnalyzer` (defined in `src/sentiment_analysis.py`) that uses VADER (with a small financial lexicon) and TextBlob to compute positive/neutral/negative scores per tweet and store them as columns.
- Missing Twitter metrics are forward-filled in the notebook (`.fillna(method='ffill')`) — consider group-wise forward-fill by stock to avoid cross-stock leakage.

**Developer notes**
- `src/` contains helpful utilities; currently only `sentiment_analysis.py` is referenced by the notebooks. The other helpers (`data_processing.py`, `feature_engineering.py`, `models.py`, `evaluation.py`) exist but are not wired into the running backend.
- To change the model, replace files in `models/` and restart the backend.

**Usage example (predict)**
```cmd
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{ \"stock_symbol\": \"INFY\" }"
```

**Troubleshooting**
- If predictions return trivial values, ensure `POST /predict` request has enough historical data available (the model expects ~50–60 days of history for full feature computation).
- If Angel One authentication fails, check environment variables and `.env` content and verify the access token / TOTP setup.

**Notebooks**
- `notebooks/01_data_preprocessing.ipynb` — cleaning and sentiment labeling
- `notebooks/02_feature_engineering_and_training.ipynb` — feature creation, model training and evaluation

**License**
This project is released under the MIT License.

