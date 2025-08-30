# ================================================================
# STREAMLIT STOCK PREDICTION APP - INDIAN STOCKS VERSION
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import os
import sys

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Indian Stock Prediction System",
    page_icon="",
    layout="wide"
)

# Import production_predictor from the same directory
try:
    from production_predictor import StockPredictionSystem
    st.success("✅ Production predictor imported successfully!")
except Exception as e:
    st.error(f"❌ Error importing production predictor: {e}")
    st.stop()

# Initialize prediction system
@st.cache_resource
def load_model():
    try:
        # Fix the path: go up ONE directory from web/ to reach the root, then into models/
        current_dir = os.path.dirname(os.path.abspath(__file__))  # web/
        root_dir = os.path.dirname(current_dir)  # project root
        models_dir = os.path.join(root_dir, "models")
        models_dir = os.path.abspath(models_dir)  # Resolve the relative path
        
        print(f"[DEBUG] Current directory: {current_dir}")
        print(f"[DEBUG] Root directory: {root_dir}")
        print(f"[DEBUG] Models directory: {models_dir}")
        
        st.info(f"[INFO] Looking for models in: {models_dir}")
        
        if not os.path.exists(models_dir):
            st.error(f"[ERROR] Models directory not found at: {models_dir}")
            return None
            
        if not os.path.exists(os.path.join(models_dir, "complete_ensemble_model.joblib")):
            st.error(f"[ERROR] Model file not found at: {os.path.join(models_dir, 'complete_ensemble_model.joblib')}")
            return None
        
        predictor = StockPredictionSystem(models_dir=models_dir)
        return predictor
    except Exception as e:
        st.error(f"[ERROR] Error loading model: {e}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"File location: {os.path.abspath(__file__)}")
        return None

# Main app
def main():
    st.title("🚀 Indian Stock Prediction System")
    st.markdown("**AI-Powered Stock Price Predictions using Ensemble Machine Learning**")
    st.info("💡 This model was trained on Indian stock data (NSE stocks)")
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.error("❌ Could not load prediction model. Please check the models folder.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header(" Stock Selection")
    
    # Indian stock symbols your model was trained on
    indian_stocks = {
        "ADANIENT.NS": "Adani Enterprises",
        "BHARTIARTL.NS": "Bharti Airtel", 
        "HCLTECH.NS": "HCL Technologies",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "INFY.NS": "Infosys",
        "SBIN.NS": "State Bank of India",
        "TATASTEEL.NS": "Tata Steel"
    }
    
    # Stock symbol input
    stock_symbol = st.sidebar.selectbox(
        "Select Indian Stock",
        options=list(indian_stocks.keys()),
        format_func=lambda x: f"{x} ({indian_stocks[x]})",
        help="These are the stocks your model was trained on"
    )
    
    # Quick buttons for popular stocks
    st.sidebar.subheader(" Quick Selection")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("INFY"):
            stock_symbol = "INFY.NS"
    with col2:
        if st.button("HDFC"):
            stock_symbol = "HDFCBANK.NS"
    
    # Date selection
    st.sidebar.subheader("📅 Date Range")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    start_date_input = st.sidebar.date_input("Start Date", value=start_date)
    end_date_input = st.sidebar.date_input("End Date", value=end_date)
    
    # Future prediction section
    st.sidebar.subheader("🔮 Future Prediction")
    st.sidebar.info("💡 Start date is needed because the AI model requires historical data to create technical indicators (moving averages, RSI, etc.)")
    
    # Days into future to predict
    future_days = st.sidebar.slider(
        "Days into Future to Predict", 
        min_value=1, 
        max_value=30, 
        value=1,
        help="How many days ahead do you want to predict? (1-30 days)"
    )
    
    # Prediction target date
    target_date = end_date + timedelta(days=future_days)
    st.sidebar.success(f"🎯 Predicting for: {target_date}")
    
    # Main content
    if stock_symbol:
        st.subheader(f"📊 Stock Data for: {stock_symbol} ({indian_stocks[stock_symbol]})")
        
        # Explain why start date is needed
        with st.expander("ℹ️ Why do I need a start date?"):
            st.write("""
            **The AI model needs historical data to make predictions because:**
            
            🎯 **Technical Indicators**: 
            - Moving averages (3-day, 5-day, 10-day, 20-day, 50-day)
            - RSI, Bollinger Bands, volatility measures
            - Price momentum and trend analysis
            
            📊 **Feature Engineering**:
            - Price changes over different time periods
            - Volume analysis and patterns
            - Market sentiment indicators
            
            ⏰ **Minimum Requirements**:
            - At least 50+ days of data for reliable features
            - More data = better feature quality = better predictions
            
            **Think of it like this**: You can't predict tomorrow's weather without knowing today's and yesterday's weather patterns!
            """)
        
        # Fetch stock data
        try:
            with st.spinner("📊 Fetching stock data..."):
                stock_data = yf.download(
                    stock_symbol,
                    start=start_date_input,
                    end=end_date_input,
                    progress=False
                )
            
            if stock_data.empty:
                st.error(f"❌ No data found for {stock_symbol}")
                st.write("**This might be due to:**")
                st.write("- Market holidays")
                st.write("- Data provider issues")
                st.write("- Try a different date range")
                return
            
                        # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                try:
                    current_price = float(stock_data['Close'].iloc[-1])
                    st.metric("Current Price", f"₹{current_price:.2f}")
                except:
                    st.metric("Current Price", "N/A")
            
            with col2:
                try:
                    if len(stock_data) > 1:
                        change = float(stock_data['Close'].iloc[-1]) - float(stock_data['Close'].iloc[-2])
                        change_pct = (change / float(stock_data['Close'].iloc[-2])) * 100
                        st.metric("Daily Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
                    else:
                        st.metric("Daily Change", "N/A")
                except:
                    st.metric("Daily Change", "N/A")
            
            with col3:
                try:
                    volume = int(stock_data['Volume'].iloc[-1])
                    st.metric("Volume", f"{volume:,}")
                except:
                    st.metric("Volume", "N/A")
            
            with col4:
                st.metric("Data Points", len(stock_data))
            
            # Price chart
            st.subheader("📈 Stock Price Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"{stock_symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction button
            if st.button("🔮 Make Prediction", type="primary"):
                with st.spinner(" Making predictions..."):
                    try:
                        # Get prediction
                        st.info("🔮 Getting AI prediction...")
                        st.write(f"Stock data shape: {stock_data.shape}")
                        st.write(f"Stock data columns: {list(stock_data.columns)}")
                        
                        # Handle multi-level column names from yfinance BEFORE calling prediction
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            st.info("🔄 Flattening multi-level columns from yfinance...")
                            stock_data = stock_data.copy()
                            stock_data.columns = stock_data.columns.get_level_values(0)
                            st.write(f"After flattening columns: {list(stock_data.columns)}")
                        
                        recommendation = predictor.get_stock_recommendation(stock_data)
                        
                        st.write(f"Recommendation result: {recommendation}")
                        st.write(f"Recommendation type: {type(recommendation)}")
                        
                        if recommendation is None:
                            st.error("❌ get_stock_recommendation returned None!")
                            st.write("This means there was an error in the prediction system.")
                            st.write("Check the console/terminal for debug information.")
                            return
                        
                        # Display results
                        st.subheader("🎯 AI Prediction Results")
                        
                        # Show prediction target
                        st.success(f"🔮 **Predicting price for: {target_date}** ({future_days} days from today)")
                        
                        # Basic recommendation
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"**Recommendation: {recommendation['recommendation']}**")
                            st.write(f"**Reasoning:** {recommendation['reasoning']}")
                        
                        with col2:
                            st.write(f"**Prediction Date:** {recommendation['price_prediction']['prediction_date']}")
                            st.write(f"**Confidence:** {recommendation['price_prediction']['confidence']:.2f}")
                        
                        # Price details
                        st.subheader("💰 Price Prediction Details")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"₹{stock_data['Close'].iloc[-1]:.2f}")
                        
                        with col2:
                            predicted_price = recommendation['price_prediction']['predicted_price']
                            st.metric("Predicted Price", f"₹{predicted_price:.2f}")
                        
                        with col3:
                            price_change = predicted_price - stock_data['Close'].iloc[-1]
                            st.metric("Expected Change", f"₹{price_change:.2f}")
                        
                        with col4:
                            change_pct = (price_change / stock_data['Close'].iloc[-1]) * 100
                            st.metric("Change %", f"{change_pct:.2f}%")
                        
                        # Model comparison
                        st.subheader("🤖 Model Predictions")
                        
                        model_predictions = recommendation['price_prediction']['model_predictions']
                        model_df = pd.DataFrame([
                            {'Model': model, 'Prediction': pred}
                            for model, pred in model_predictions.items()
                        ])
                        
                        fig = px.bar(
                            model_df,
                            x='Model',
                            y='Prediction',
                            title="Individual Model Predictions",
                            color='Model'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Direction
                        st.subheader("📊 Direction Prediction")
                        
                        direction = recommendation['direction_prediction']
                        
                        if direction['direction'] == "UP":
                            st.success(f"📈 **Direction: {direction['direction']}**")
                        else:
                            st.error(f"📉 **Direction: {direction['direction']}**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Expected Change:** {direction['change_percent']:.2f}%")
                            st.write(f"**Current Price:** ₹{direction['current_price']:.2f}")
                        
                        with col2:
                            st.write(f"**Predicted Price:** ₹{direction['predicted_price']:.2f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {e}")
                        st.write("**Debug info:**")
                        st.write(f"- Stock data shape: {stock_data.shape}")
                        st.write(f"- Error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            st.write("**Try these solutions:**")
            st.write("1. Check your internet connection")
            st.write("2. Try a different date range")
            st.write("3. The stock might be on holiday")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Streamlit, Plotly, and Ensemble Machine Learning**")
    st.markdown("**Trained on Indian Stock Market Data (NSE)**")

if __name__ == "__main__":
    main()