import os
from SmartApi import SmartConnect
import pyotp
from dotenv import load_dotenv

# ==== Credentials ====
API_KEY = "KQZrjKDX"
CLIENT_CODE = "AABX480655"
MPIN = "6969"  # Your 4-digit SmartAPI MPIN
TOTP_SECRET = "WBWETMN5YX4RM7WNWHADBJQU6U"

# ==== Initialize SmartConnect ====
obj = SmartConnect(API_KEY)

# ==== Generate TOTP ====
totp = pyotp.TOTP(TOTP_SECRET).now()
print(f"Generated TOTP: {totp}")

# ==== Login using MPIN + TOTP (positional arguments) ====
data = obj.generateSession(CLIENT_CODE, MPIN, totp)

ACCESS_TOKEN = "Bearer " + data['data']['jwtToken']
FEED_TOKEN = data['data']['feedToken']

print("✅ Login successful!")
print("Access Token:", ACCESS_TOKEN)
print("Feed Token:", FEED_TOKEN)

# ==== Save tokens to .env robustly ====
env_path = ".env"

# Create .env if it doesn't exist
if not os.path.exists(env_path):
    with open(env_path, "w") as f:
        f.write("")

# Load existing env variables
load_dotenv(env_path)

# Write / update ACCESS_TOKEN and FEED_TOKEN
def save_env(key, value):
    # Read all lines
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    
    # Check if key exists, replace it
    found = False
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")
    
    # Write back
    with open(env_path, "w") as f:
        f.writelines(lines)

# Save tokens
save_env("ACCESS_TOKEN", ACCESS_TOKEN)
save_env("FEED_TOKEN", FEED_TOKEN)
print("✅ Tokens saved to .env successfully!")
