import sys
import os

# Simulate the exact import path that Streamlit uses
print("=== SIMULATING STREAMLIT IMPORT AND MODEL LOADING ===")

# Add the web directory to the path (same as Streamlit)
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))
print(f"[DEBUG] Added web directory to path: {os.path.join(os.path.dirname(__file__), 'web')}")

try:
    # Import the predictor (same as Streamlit)
    print("\n1. Testing import...")
    from production_predictor import StockPredictionSystem
    print("[SUCCESS] Import successful!")
    
    # Load the model (same as Streamlit)
    print("\n2. Testing model loading...")
    # Use the exact path calculation from the Streamlit app
    current_dir = os.path.dirname(os.path.abspath(__file__))  # project root
    web_dir = os.path.join(current_dir, "web")  # simulate being in web/
    root_dir = os.path.dirname(web_dir)  # project root
    models_dir = os.path.join(root_dir, "models")
    models_dir = os.path.abspath(models_dir)
    
    print(f"[DEBUG] Models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"[ERROR] Models directory not found!")
        sys.exit(1)
        
    if not os.path.exists(os.path.join(models_dir, "complete_ensemble_model.joblib")):
        print(f"[ERROR] Model file not found!")
        sys.exit(1)
    
    predictor = StockPredictionSystem(models_dir=models_dir)
    print("[SUCCESS] Model loading successful!")
    
    # Test with minimal data
    print("\n3. Testing prediction...")
    import pandas as pd
    import numpy as np
    
    # Create minimal test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    stock_data = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'High': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0],
        'Low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=dates)
    
    print(f"[DEBUG] Test data shape: {stock_data.shape}")
    
    # Test the prediction
    recommendation = predictor.get_stock_recommendation(stock_data)
    
    if recommendation is None:
        print("[ERROR] get_stock_recommendation returned None!")
        print("This is the exact error you're seeing in Streamlit!")
    else:
        print("[SUCCESS] Prediction successful!")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Reasoning: {recommendation['reasoning']}")
    
    print("\n[SUCCESS] All tests passed! The Streamlit app should work now.")
    
except Exception as e:
    print(f"\n[ERROR] ERROR OCCURRED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
