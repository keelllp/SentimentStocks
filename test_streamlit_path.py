import os

# Simulate exactly what the Streamlit app does
print("=== SIMULATING STREAMLIT PATH CALCULATION ===")

# Simulate being in web/streamlit_app.py
# The file is located at: web/streamlit_app.py
# We want to find: models/ (which is in the parent directory)

# This is the exact logic from the Streamlit app
current_dir = os.path.dirname(os.path.abspath(__file__))  # current directory (where this script is)
print(f"[DEBUG] Current directory: {current_dir}")

# Since we're running from the project root, let's simulate being in the web/ directory
web_dir = os.path.join(current_dir, "web")
print(f"[DEBUG] Web directory: {web_dir}")

# Now simulate the Streamlit app path calculation
current_dir = web_dir  # This is what __file__ would be in streamlit_app.py
root_dir = os.path.dirname(current_dir)  # project root
models_dir = os.path.join(root_dir, "models")
models_dir = os.path.abspath(models_dir)  # Resolve the relative path

print(f"[DEBUG] Current directory (web): {current_dir}")
print(f"[DEBUG] Root directory: {root_dir}")
print(f"[DEBUG] Models directory: {models_dir}")

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

print("\n=== END OF TEST ===")
