import sys
import os

# Add the web directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

try:
    print("=== DEBUGGING PREDICTION SYSTEM ===")
    
    # Test import
    print("\n1. Testing import...")
    from production_predictor import StockPredictionSystem
    print("✅ Import successful!")
    
    # Test model loading
    print("\n2. Testing model loading...")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    print(f"Models directory: {models_dir}")
    
    predictor = StockPredictionSystem(models_dir=models_dir)
    print("✅ Model loading successful!")
    
    # Test with real stock data format (similar to what yfinance returns)
    print("\n3. Testing with stock data...")
    import pandas as pd
    import numpy as np
    
    # Create realistic stock data (similar to yfinance output)
    dates = pd.date_range(start='2024-01-01', end='2024-01-20', freq='D')
    stock_data = pd.DataFrame({
        'Open': [100.0, 101.5, 102.3, 103.1, 104.2, 105.0, 106.1, 107.2, 108.0, 109.1, 
                 110.0, 111.2, 112.1, 113.0, 114.2, 115.0, 116.1, 117.0, 118.1, 119.0],
        'High': [105.0, 106.5, 107.3, 108.1, 109.2, 110.0, 111.1, 112.2, 113.0, 114.1,
                 115.0, 116.2, 117.1, 118.0, 119.2, 120.0, 121.1, 122.0, 123.1, 124.0],
        'Low': [95.0, 96.5, 97.3, 98.1, 99.2, 100.0, 101.1, 102.2, 103.0, 104.1,
                105.0, 106.2, 107.1, 108.0, 109.2, 110.0, 111.1, 112.0, 113.1, 114.0],
        'Close': [102.0, 103.5, 104.3, 105.1, 106.2, 107.0, 108.1, 109.2, 110.0, 111.1,
                  112.0, 113.2, 114.1, 115.0, 116.2, 117.0, 118.1, 119.0, 120.1, 121.0],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                   2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900]
    }, index=dates)
    
    print(f"Stock data shape: {stock_data.shape}")
    print(f"Stock data columns: {list(stock_data.columns)}")
    print(f"Stock data types: {stock_data.dtypes.to_dict()}")
    
    # Test feature creation step by step
    print("\n4. Testing feature creation...")
    features = predictor.create_features(stock_data)
    if features is not None:
        print(f"✅ Feature creation successful! Shape: {features.shape}")
        print(f"Feature columns: {list(features.columns)}")
    else:
        print("❌ Feature creation failed!")
        sys.exit(1)
    
    # Test feature preparation
    print("\n5. Testing feature preparation...")
    prepared_features = predictor.prepare_features(stock_data)
    if prepared_features is not None:
        print(f"✅ Feature preparation successful! Shape: {prepared_features.shape}")
    else:
        print("❌ Feature preparation failed!")
        sys.exit(1)
    
    # Test price prediction
    print("\n6. Testing price prediction...")
    price_pred = predictor.predict_price(stock_data)
    if price_pred is not None:
        print(f"✅ Price prediction successful!")
        print(f"Ensemble prediction: {price_pred['ensemble_prediction']}")
        print(f"Model predictions: {price_pred['model_predictions']}")
    else:
        print("❌ Price prediction failed!")
        sys.exit(1)
    
    # Test final recommendation
    print("\n7. Testing final recommendation...")
    recommendation = predictor.get_stock_recommendation(stock_data)
    if recommendation:
        print("✅ Recommendation successful!")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Reasoning: {recommendation['reasoning']}")
        print(f"Predicted price: {recommendation['price_prediction']['predicted_price']}")
        print(f"Current price: {recommendation['price_prediction']['current_price']}")
    else:
        print("❌ Recommendation failed!")
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED! The prediction system is working correctly.")
    
except Exception as e:
    print(f"\n❌ ERROR OCCURRED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
