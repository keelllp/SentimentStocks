import os
from SmartApi import SmartConnect
import pyotp
from dotenv import load_dotenv

# ==== Credentials ====
API_KEY = "UsOGGTK1"
CLIENT_CODE = "AABX480655"
MPIN = "6969"
TOTP_SECRET = "WBWETMN5YX4RM7WNWHADBJQU6U"

# ==== Initialize SmartConnect ====
obj = SmartConnect(API_KEY)

# ==== Generate TOTP ====
totp = pyotp.TOTP(TOTP_SECRET).now()

# ==== Login ====
data = obj.generateSession(CLIENT_CODE, MPIN, totp)

ACCESS_TOKEN = data['data']['jwtToken']
FEED_TOKEN = data['data']['feedToken']
REFRESH_TOKEN = data['data']['refreshToken']

print("✅ Login successful!")

# ==== Save to .env ====
env_path = ".env"

def save_env(key, value):
    lines = []

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    found = False

    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}\n"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

save_env("ACCESS_TOKEN", ACCESS_TOKEN)
save_env("FEED_TOKEN", FEED_TOKEN)
save_env("REFRESH_TOKEN", REFRESH_TOKEN)

print("✅ Tokens saved successfully!")