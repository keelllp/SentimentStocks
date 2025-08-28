import sys
import os

# Add the web directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

try:
    print("=== TESTING WITH YFINANCE-LIKE DATA ===")
    
    # Import the predictor
    from production_predictor import StockPredictionSystem
    print("✅ Import successful!")
    
    # Load the model
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    predictor = StockPredictionSystem(models_dir=models_dir)
    print("✅ Model loaded!")
    
    # Simulate yfinance data exactly
    import pandas as pd
    import numpy as np
    
    # Create data that mimics yfinance output exactly
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    
    # yfinance typically returns float64 for prices and int64 for volume
    stock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)).astype('float64'),
        'High': np.random.uniform(200, 300, len(dates)).astype('float64'),
        'Low': np.random.uniform(50, 100, len(dates)).astype('float64'),
        'Close': np.random.uniform(100, 200, len(dates)).astype('float64'),
        'Volume': np.random.randint(1000, 10000, len(dates)).astype('int64')
    }, index=dates)
    
    print(f"Simulated yfinance data:")
    print(f"- Shape: {stock_data.shape}")
    print(f"- Columns: {list(stock_data.columns)}")
    print(f"- Types: {stock_data.dtypes.to_dict()}")
    print(f"- Index type: {type(stock_data.index)}")
    print(f"- Sample data:\n{stock_data.head()}")
    
    # Test the prediction step by step with error handling
    print("\n=== TESTING PREDICTION PIPELINE ===")
    
    try:
        print("\n1. Testing feature creation...")
        features = predictor.create_features(stock_data)
        if features is not None:
            print(f"✅ Feature creation successful! Shape: {features.shape}")
        else:
            print("❌ Feature creation returned None!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Feature creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        print("\n2. Testing feature preparation...")
        prepared_features = predictor.prepare_features(stock_data)
        if prepared_features is not None:
            print(f"✅ Feature preparation successful! Shape: {prepared_features.shape}")
        else:
            print("❌ Feature preparation returned None!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        print("\n3. Testing price prediction...")
        price_pred = predictor.predict_price(stock_data)
        if price_pred is not None:
            print(f"✅ Price prediction successful!")
            print(f"Ensemble prediction: {price_pred['ensemble_prediction']}")
        else:
            print("❌ Price prediction returned None!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Price prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        print("\n4. Testing final recommendation...")
        recommendation = predictor.get_stock_recommendation(stock_data)
        if recommendation is not None:
            print("✅ Recommendation successful!")
            print(f"Recommendation: {recommendation['recommendation']}")
            print(f"Reasoning: {recommendation['reasoning']}")
        else:
            print("❌ Recommendation returned None!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED WITH YFINANCE-LIKE DATA!")
    
except Exception as e:
    print(f"\n❌ ERROR OCCURRED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
