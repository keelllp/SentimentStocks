# Stock Price Prediction System

A comprehensive machine learning system for predicting stock prices using advanced ensemble models, technical indicators, and sentiment analysis.

## 🏗️ Project Structure

```
stock-prediction-system/
├── src/                    # Core source code
│   ├── data_processing.py  # Data preprocessing utilities
│   ├── feature_engineering.py # Feature creation and engineering
│   ├── models.py          # Model definitions and training
│   ├── evaluation.py      # Model evaluation metrics
│   └── sentiment_analysis.py # Sentiment analysis tools
├── web/                   # Web applications
│   ├── production_predictor.py # Production prediction system
│   ├── improved_model_training.py # Enhanced model training
│   ├── streamlit_app.py   # Streamlit web interface
│   ├── app.py            # Flask web API
│   └── templates/        # HTML templates
├── models/               # Trained model files
├── data/                # Raw data files
├── processed_data/      # Processed datasets
├── notebooks/           # Jupyter notebooks for analysis
├── results/             # Model results and outputs
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Improved Models
```bash
cd web
python improved_model_training.py
```

### 3. Run Production Predictor
```bash
python production_predictor.py
```

### 4. Launch Web Interface
```bash
# Streamlit app
streamlit run streamlit_app.py

# Flask API
python app.py
```

## 🔧 Model Improvements

The system includes several enhancements for better accuracy:

- **Advanced Feature Engineering**: Technical indicators, volatility measures, momentum features
- **Ensemble Methods**: XGBoost, LightGBM, Random Forest with optimized weights
- **Time-Series Cross-Validation**: Proper validation for time-series data
- **Feature Selection**: Advanced feature selection using multiple methods
- **Hyperparameter Optimization**: Automated tuning for optimal performance

## 📊 Current Performance

- **Direction Accuracy**: 60-75% (improved from 24-62%)
- **MAPE**: 2-8% (improved from 1-80%)
- **Better Generalization**: Across different stocks and market conditions

## 🎯 Key Features

- Real-time stock data fetching with yfinance
- Advanced technical indicators and market features
- Sentiment analysis from social media
- Ensemble prediction with confidence scoring
- Web interface for easy interaction
- RESTful API for integration

## 📈 Usage Examples

### Basic Prediction
```python
from web.production_predictor import StockPredictionSystem

predictor = StockPredictionSystem()
prediction = predictor.predict_price(stock_data)
print(f"Predicted price: ${prediction['ensemble_prediction']:.2f}")
```

### Model Training
```python
from web.improved_model_training import ImprovedStockPredictor

trainer = ImprovedStockPredictor()
models = trainer.train_models(X, y, optimize=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
