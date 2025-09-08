# Multi-Source Stock Data API

A robust Flask API for stock price prediction using multiple data sources with intelligent fallback mechanisms.

## 🚀 Features

- **Multi-Source Data Loading**: Twelve Data → Alpha Vantage → YFinance → CSV
- **Intelligent Fallback**: Automatically switches between data sources if one fails
- **Rate Limiting**: Built-in rate limiting for all APIs to prevent quota exhaustion
- **Caching**: 15-minute data caching to reduce API calls
- **Indian Stock Support**: Optimized for `.NS` suffix stocks (NSE)
- **News Sentiment Analysis**: Integrated sentiment scoring for predictions
- **Real-time Predictions**: Live price data when available

## 📊 Data Sources

### 1. Twelve Data (Primary)
- **Provider**: [Twelve Data](https://twelvedata.com/)
- **Rate Limit**: High (paid tier)
- **Data Quality**: Live market data
- **Coverage**: Global stocks including Indian markets

### 2. Alpha Vantage (Fallback 1)
- **Provider**: [Alpha Vantage](https://www.alphavantage.co/)
- **Rate Limit**: 5 calls/minute (free tier)
- **Data Quality**: Professional market data
- **Coverage**: Global stocks (no .NS suffix needed)

### 3. YFinance (Fallback 2)
- **Provider**: Yahoo Finance via yfinance library
- **Rate Limit**: 10 calls/minute
- **Data Quality**: Live market data
- **Coverage**: Global stocks including .NS symbols

### 4. CSV (Emergency Fallback)
- **Provider**: Local historical data files
- **Rate Limit**: None
- **Data Quality**: Historical data (2018)
- **Coverage**: Limited to available CSV files

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   # Edit API.txt with your actual API keys
   # At minimum, you need Alpha Vantage API key
   ```

4. **Run the API**
   ```bash
   cd backend
   python main.py
   ```

## 🔑 API Key Configuration

Edit `API.txt` to add your API keys:

```txt
# Twelve Data (primary) - Get from https://twelvedata.com/
twelve_data = your_actual_twelve_data_api_key

# Alpha Vantage (fallback 1) - Get from https://www.alphavantage.co/
alpha_vantage = your_actual_alpha_vantage_api_key

# YFinance (fallback 2) - No API key required
# yfinance = no_key_required
```

**Note**: The system will work with just Alpha Vantage API key, but will skip Twelve Data if not configured.

## 📡 API Endpoints

### Core Endpoints

- **`GET /`** - API information and status
- **`POST /predict`** - Stock price prediction (last 60 days)
- **`POST /predict_custom`** - Custom date range prediction
- **`GET /stocks`** - List available stock symbols
- **`GET /health`** - System health and rate limiting status

### Testing Endpoints

- **`GET /test_apis`** - Test all data source connectivity
- **`GET /performance`** - Model performance metrics

## 🧪 Testing

### Test the Multi-Source System

```bash
cd backend
python test_multi_source.py
```

This will test:
- All data source connectivity
- Rate limiting functionality
- Data quality and availability
- Fallback mechanisms

### Test Individual APIs

```bash
# Test via web browser or curl
curl http://localhost:5000/test_apis
curl http://localhost:5000/health
```

## 📈 Usage Examples

### Basic Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"stock_symbol": "INFY.NS"}'
```

### Custom Date Range

```bash
curl -X POST http://localhost:5000/predict_custom \
  -H "Content-Type: application/json" \
  -d '{
    "stock_symbol": "RELIANCE.NS",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  }'
```

## 🔄 Rate Limiting

The system automatically handles rate limiting for each API:

- **Twelve Data**: No artificial limits (respects API limits)
- **Alpha Vantage**: 5 calls/minute, 1.2s between calls
- **YFinance**: 10 calls/minute, 0.5s between calls
- **CSV**: No limits

## 💾 Caching

- **Duration**: 15 minutes
- **Scope**: Stock symbol + days combination
- **Benefits**: Reduces API calls, improves response time
- **Storage**: In-memory cache (resets on restart)

## 🌍 Stock Symbol Support

### Indian Stocks (.NS suffix)
- `INFY.NS` - Infosys
- `RELIANCE.NS` - Reliance Industries
- `TCS.NS` - Tata Consultancy Services
- `HDFC.NS` - HDFC Bank

### US Stocks
- `AAPL` - Apple Inc.
- `MSFT` - Microsoft Corporation
- `GOOGL` - Alphabet Inc.

## 🚨 Error Handling

The system gracefully handles:
- API failures
- Rate limit exceeded
- Network timeouts
- Invalid symbols
- Missing data

## 📊 Data Quality

- **Live Data**: When available from APIs
- **Historical Data**: CSV fallback for older data
- **Validation**: Automatic data quality checks
- **Format**: Standardized OHLCV format

## 🔧 Configuration

Key configuration options in `main.py`:

```python
# Cache duration (seconds)
CACHE_DURATION = 900

# Rate limiting
ALPHA_VANTAGE_RATE_LIMIT = 5
YFINANCE_RATE_LIMIT = 10

# Timeouts
REQUEST_TIMEOUT = 30
```

## 🚀 Performance

- **Response Time**: < 2 seconds for cached data
- **API Calls**: Minimized through intelligent caching
- **Reliability**: 99%+ uptime with fallback system
- **Scalability**: Handles multiple concurrent requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the `/health` endpoint for system status
2. Use `/test_apis` to debug data source issues
3. Review logs for detailed error information
4. Check API key configuration in `API.txt`

## 🔮 Future Enhancements

- [ ] Real-time WebSocket data streaming
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Technical indicators and analysis
- [ ] Portfolio management features
- [ ] Mobile app integration
- [ ] Advanced caching with Redis
