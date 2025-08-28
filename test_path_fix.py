import os

# Test the path calculation that Streamlit will use
print("=== TESTING PATH CALCULATION ===")

# Get the current file's directory (simulating web/streamlit_app.py)
current_file = __file__
current_dir = os.path.dirname(os.path.abspath(current_file))
print(f"Current file: {current_file}")
print(f"Current directory: {current_dir}")

# Calculate the models directory path (same logic as Streamlit - CORRECTED)
current_dir = os.path.dirname(os.path.abspath(current_file))  # current directory
root_dir = os.path.dirname(current_dir)  # project root
models_dir = os.path.join(root_dir, "models")
models_dir = os.path.abspath(models_dir)  # Resolve the relative path
print(f"Calculated models directory: {models_dir}")

# Check if it exists
if os.path.exists(models_dir):
    print(f"[SUCCESS] Models directory exists!")
    
    # Check for the model file
    model_file = os.path.join(models_dir, "complete_ensemble_model.joblib")
    if os.path.exists(model_file):
        print(f"[SUCCESS] Model file exists!")
        print(f"Model file size: {os.path.getsize(model_file)} bytes")
    else:
        print(f"[ERROR] Model file not found!")
else:
    print(f"[ERROR] Models directory not found!")

# Also test the old path calculation
old_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(current_file))), "models")
print(f"Old path calculation: {old_models_dir}")
if os.path.exists(old_models_dir):
    print(f"[SUCCESS] Old path also exists!")
else:
    print(f"[ERROR] Old path doesn't exist!")

print("\n=== END OF TEST ===")
