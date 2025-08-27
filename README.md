# Stock Tweets Sentiment Analysis and Price Prediction

This project analyzes Twitter sentiment data for Indian stocks and predicts stock prices using machine learning models (XGBoost and LightGBM).

## Project Overview

The project combines:
- **Twitter Sentiment Analysis**: Analyzes tweets related to stock symbols
- **Stock Price Data**: Historical stock price data from Yahoo Finance
- **Machine Learning Models**: Price prediction using sentiment features and technical indicators
- **Web Frontend**: Interactive dashboard for predictions and analysis

## Project Structure

```
├── data/                          # Data files
│   ├── stock_data/               # Historical stock prices
│   ├── twitter_data/             # Scraped Twitter data
│   └── twitter_data_new/         # Updated Twitter data with sentiment
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   └── 02_model_training.ipynb
├── src/                          # Source code
│   ├── data_processing.py        # Data processing utilities
│   ├── sentiment_analysis.py     # Sentiment analysis functions
│   ├── feature_engineering.py    # Feature engineering
│   ├── models.py                 # ML models (XGBoost, LightGBM)
│   └── evaluation.py             # Model evaluation
├── web/                          # Web frontend
│   ├── app.py                    # Flask backend
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS, JS, images
│   └── streamlit_app.py          # Streamlit alternative
├── models/                       # Trained models
├── results/                      # Results and visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

3. **Run Data Preprocessing**:
   ```bash
   jupyter notebook notebooks/01_data_preprocessing.ipynb
   ```

4. **Train Models**:
   ```bash
   jupyter notebook notebooks/02_model_training.ipynb
   ```

5. **Run Web Application**:
   ```bash
   # Flask backend
   python web/app.py
   
   # Streamlit frontend
   streamlit run web/streamlit_app.py
   ```

## Usage

1. **Data Preprocessing**: Run `01_data_preprocessing.ipynb` to clean and prepare data
2. **Model Training**: Run `02_model_training.ipynb` to train XGBoost and LightGBM models
3. **Web Interface**: Access the web application for interactive predictions

## Features

- **Sentiment Analysis**: VADER and TextBlob sentiment scoring
- **Feature Engineering**: Technical indicators, sentiment features, and time-based features
- **Models**: XGBoost and LightGBM for price prediction
- **Evaluation**: RMSE, MAE, R², and visualization metrics
- **Web Interface**: Interactive dashboard for predictions

## Data Sources

- **Stocks**: NSE-listed stocks (BHARTIARTL, INFY, RELIANCE, etc.)
- **Twitter**: Public tweets with stock hashtags
- **Time Period**: 2018-2023

## Contributing

Feel free to contribute by improving models, adding new features, or enhancing the analysis.

## License

This project is for educational purposes.
