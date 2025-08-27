"""
Streamlit frontend for Stock Price Prediction Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-result {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
models = {}
scaler = None
feature_names = None

def load_models():
    """Load trained models and scaler"""
    global models, scaler, feature_names
    
    models_dir = "../models"
    
    try:
        # Load XGBoost model
        xgb_path = os.path.join(models_dir, "stock_predictor_xgboost.pkl")
        if os.path.exists(xgb_path):
            models['xgboost'] = joblib.load(xgb_path)
            st.success("✅ XGBoost model loaded successfully")
        
        # Load LightGBM model
        lgb_path = os.path.join(models_dir, "stock_predictor_lightgbm.pkl")
        if os.path.exists(lgb_path):
            models['lightgbm'] = joblib.load(lgb_path)
            st.success("✅ LightGBM model loaded successfully")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, "stock_predictor_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.success("✅ Scaler loaded successfully")
        
        # Load feature names
        features_path = os.path.join(models_dir, "stock_predictor_features.pkl")
        if os.path.exists(features_path):
            feature_names = joblib.load(features_path)
            st.success("✅ Feature names loaded successfully")
        
        return len(models) > 0 and scaler is not None and feature_names is not None
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return False

def get_available_stocks():
    """Get list of available stocks"""
    try:
        stock_dir = "../data/stock_data"
        if not os.path.exists(stock_dir):
            return []
        
        stock_files = [f.replace('.csv', '') for f in os.listdir(stock_dir) 
                      if f.endswith('.csv')]
        return stock_files
        
    except Exception as e:
        st.error(f"Error loading stocks: {e}")
        return []

def load_stock_data(stock_symbol):
    """Load stock data for visualization"""
    try:
        stock_path = f"../data/stock_data/{stock_symbol}.csv"
        if not os.path.exists(stock_path):
            return None
        
        df = pd.read_csv(stock_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
        
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return None

def prepare_features(stock_data):
    """Prepare features for prediction"""
    try:
        # This is a simplified feature preparation
        # In a real application, you would use the same feature engineering pipeline
        
        features = {}
        
        # Basic price features
        features['Close'] = stock_data['Close'].iloc[-1]
        features['Open'] = stock_data['Open'].iloc[-1]
        features['High'] = stock_data['High'].iloc[-1]
        features['Low'] = stock_data['Low'].iloc[-1]
        
        # Moving averages
        features['ma_5'] = stock_data['Close'].rolling(5).mean().iloc[-1]
        features['ma_20'] = stock_data['Close'].rolling(20).mean().iloc[-1]
        
        # Price changes
        features['price_change_1d'] = stock_data['Close'].pct_change().iloc[-1]
        features['price_change_5d'] = stock_data['Close'].pct_change(5).iloc[-1]
        
        # Volatility
        features['volatility_5d'] = stock_data['Close'].pct_change().rolling(5).std().iloc[-1]
        
        # Volume features (if available)
        if 'Volume' in stock_data.columns:
            features['volume'] = stock_data['Volume'].iloc[-1]
            features['volume_ma_5'] = stock_data['Volume'].rolling(5).mean().iloc[-1]
        else:
            features['volume'] = 1000  # Default value
            features['volume_ma_5'] = 1000
        
        # Time features
        today = datetime.now()
        features['day_of_week'] = today.weekday()
        features['month'] = today.month
        features['quarter'] = (today.month - 1) // 3 + 1
        
        # Create feature vector
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        return np.array(feature_vector).reshape(1, -1)
        
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None

def make_prediction(features, model_type):
    """Make prediction using loaded models"""
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        if model_type == 'xgboost' and 'xgboost' in models:
            prediction = models['xgboost'].predict(features_scaled)[0]
        elif model_type == 'lightgbm' and 'lightgbm' in models:
            prediction = models['lightgbm'].predict(features_scaled)[0]
        elif model_type == 'ensemble' and len(models) > 1:
            # Ensemble prediction
            predictions = []
            for model in models.values():
                pred = model.predict(features_scaled)[0]
                predictions.append(pred)
            prediction = np.mean(predictions)
        else:
            # Use first available model
            model_name = list(models.keys())[0]
            prediction = models[model_name].predict(features_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative price
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_stock_chart(df):
    """Create interactive stock price chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Stock Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#74b9ff', width=2)
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if len(df) >= 5:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'].rolling(5).mean(),
                mode='lines',
                name='MA 5',
                line=dict(color='#e17055', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    if len(df) >= 20:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'].rolling(20).mean(),
                mode='lines',
                name='MA 20',
                line=dict(color='#00b894', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Volume chart
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(116, 185, 255, 0.5)'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title='Stock Price and Volume',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Stock Price Prediction</h1>
        <p>AI-powered stock price prediction using sentiment analysis and technical indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Model loading
    st.sidebar.header("Model Status")
    if st.sidebar.button("Load Models"):
        with st.spinner("Loading models..."):
            models_loaded = load_models()
            if models_loaded:
                st.sidebar.success("All models loaded successfully!")
            else:
                st.sidebar.error("Failed to load some models")
    
    # Check model status
    models_loaded = len(models) > 0 and scaler is not None and feature_names is not None
    if models_loaded:
        st.sidebar.success("✅ Models Ready")
    else:
        st.sidebar.warning("⚠️ Models Not Loaded")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Stock Selection")
        
        # Get available stocks
        available_stocks = get_available_stocks()
        
        if available_stocks:
            selected_stock = st.selectbox(
                "Choose a stock:",
                available_stocks,
                index=0 if available_stocks else None
            )
            
            if selected_stock:
                # Load stock data
                stock_data = load_stock_data(selected_stock)
                
                if stock_data is not None:
                    # Display stock info
                    current_price = stock_data['Close'].iloc[-1]
                    price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                    price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
                    
                    # Metrics
                    col1_1, col1_2, col1_3 = st.columns(3)
                    
                    with col1_1:
                        st.metric(
                            "Current Price",
                            f"₹{current_price:.2f}",
                            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                        )
                    
                    with col1_2:
                        st.metric(
                            "52 Week High",
                            f"₹{stock_data['High'].max():.2f}"
                        )
                    
                    with col1_3:
                        st.metric(
                            "52 Week Low",
                            f"₹{stock_data['Low'].min():.2f}"
                        )
                    
                    # Stock chart
                    st.plotly_chart(create_stock_chart(stock_data), use_container_width=True)
                    
                else:
                    st.error("Failed to load stock data")
        else:
            st.warning("No stock data available")
    
    with col2:
        st.header("🔮 Price Prediction")
        
        if not models_loaded:
            st.warning("Please load models first to make predictions")
            return
        
        # Prediction form
        with st.form("prediction_form"):
            # Stock selection (if not already selected)
            if 'selected_stock' not in locals():
                selected_stock = st.selectbox(
                    "Select stock for prediction:",
                    available_stocks if available_stocks else []
                )
            
            # Model selection
            model_type = st.selectbox(
                "Choose model:",
                ['ensemble', 'xgboost', 'lightgbm'],
                index=0,
                help="Ensemble combines both models for better accuracy"
            )
            
            # Prediction button
            predict_button = st.form_submit_button("🚀 Predict Price")
            
            if predict_button and selected_stock:
                with st.spinner("Making prediction..."):
                    # Load stock data
                    stock_data = load_stock_data(selected_stock)
                    
                    if stock_data is not None:
                        # Prepare features
                        features = prepare_features(stock_data)
                        
                        if features is not None:
                            # Make prediction
                            prediction = make_prediction(features, model_type)
                            
                            if prediction is not None:
                                # Display results
                                current_price = stock_data['Close'].iloc[-1]
                                price_change = prediction - current_price
                                price_change_pct = (price_change / current_price) * 100
                                
                                st.markdown(f"""
                                <div class="prediction-result">
                                    <h3>🎯 Prediction Results</h3>
                                    <h2>₹{prediction:.2f}</h2>
                                    <p>Current: ₹{current_price:.2f}</p>
                                    <p>Change: ₹{price_change:+.2f} ({price_change_pct:+.2f}%)</p>
                                    <p><small>Model: {model_type.upper()} | Date: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}</small></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Additional insights
                                st.subheader("📈 Insights")
                                
                                if price_change > 0:
                                    st.success("📈 Bullish prediction - Stock expected to rise")
                                else:
                                    st.error("📉 Bearish prediction - Stock expected to fall")
                                
                                # Model confidence (simplified)
                                if model_type == 'ensemble':
                                    st.info("🎯 Ensemble model provides more stable predictions")
                                else:
                                    st.info(f"🎯 Using {model_type.upper()} model for prediction")
                                
                            else:
                                st.error("Failed to make prediction")
                        else:
                            st.error("Failed to prepare features")
                    else:
                        st.error("Failed to load stock data")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit, XGBoost, and LightGBM</p>
        <p>Data: Yahoo Finance | Sentiment: Twitter Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
