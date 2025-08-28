import sys
import os

# Simulate the exact import path that Streamlit uses
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

try:
    print("=== SIMULATING STREAMLIT APP CALL ===")
    
    # Import the predictor (same as Streamlit)
    from production_predictor import StockPredictionSystem
    print("✅ Import successful!")
    
    # Load the model (same as Streamlit)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    print(f"Models directory: {models_dir}")
    
    predictor = StockPredictionSystem(models_dir=models_dir)
    print("✅ Model loaded!")
    
    # Now simulate the exact error scenario
    print("\n=== TESTING THE EXACT ERROR SCENARIO ===")
    
    # Create a minimal stock data (similar to what yfinance might return)
    import pandas as pd
    import numpy as np
    
    # Create very simple data (just 5 rows to test)
    dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
    stock_data = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'High': [105.0, 106.0, 107.0, 108.0, 109.0],
        'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)
    
    print(f"Test stock data:")
    print(f"- Shape: {stock_data.shape}")
    print(f"- Columns: {list(stock_data.columns)}")
    print(f"- Types: {stock_data.dtypes.to_dict()}")
    print(f"- Sample:\n{stock_data}")
    
    # Now call the exact method that's failing in Streamlit
    print(f"\n=== CALLING get_stock_recommendation ===")
    
    try:
        recommendation = predictor.get_stock_recommendation(stock_data)
        
        if recommendation is None:
            print("❌ get_stock_recommendation returned None!")
            print("This is the exact error you're seeing in Streamlit!")
        else:
            print("✅ get_stock_recommendation successful!")
            print(f"Recommendation: {recommendation['recommendation']}")
            print(f"Reasoning: {recommendation['reasoning']}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== END OF TEST ===")
    
except Exception as e:
    print(f"\n❌ ERROR OCCURRED: {e}")
    import traceback
    traceback.print_exc()
