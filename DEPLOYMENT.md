# Deployment Guide

## What's included

```
SentimentStocks/
├── backend/Dockerfile          ← Python/Flask image
├── frontend/Dockerfile         ← Node build → nginx image
├── frontend/nginx.conf         ← Serves React + proxies /api/* to Flask
├── docker-compose.yml          ← Runs both containers together
├── .env.example                ← Template for secrets
└── .dockerignore               ← Keeps images lean
```

---

## Step 1 — Create your `.env` file

Copy `.env.example` to `.env` and fill in your Angel One credentials:

```
API_KEY=your_angel_one_api_key
CLIENT_CODE=your_client_id
MPIN=your_4_digit_mpin
TOTP_SECRET=your_base32_totp_secret
FLASK_SECRET_KEY=<generate below>
```

Generate a secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

> `.env` is gitignored and never committed. Keep it safe.

---

## Step 2 — Stage the model files in git

The trained LightGBM models are untracked by default. Add them before deploying:

```bash
git add models/direction_model.pkl models/magnitude_model.pkl \
        models/label_encoder.pkl models/model_metadata.json
git add processed_data/.gitkeep
git commit -m "Add LightGBM models and deployment files"
git push
```

---

## Step 3A — Local (Docker Desktop)

Make sure Docker Desktop is running, then:

```bash
docker compose up --build
```

The app is at **http://localhost**

Useful commands:
```bash
docker compose logs -f backend          # tail backend logs
docker compose logs -f frontend         # tail nginx logs
docker compose down                     # stop everything
docker compose up --build --no-cache    # force full rebuild
```

---

## Step 3B — Free Cloud Deployment

### Architecture note

The `docker-compose.yml` runs both containers on the same host and uses nginx to proxy `/api/*` to the Flask backend internally (`http://backend:5000`). On cloud platforms where frontend and backend are **separate services with different URLs**, the nginx proxy won't reach the backend. In that case, set `REACT_APP_API_URL` to the backend's full public URL so the browser calls it directly.

---

### Option 1 — Railway (recommended, no spindown)

Railway gives $5 free credit/month which is enough for a small always-on app.

1. Push your repo to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Create **two services** from the same repo:

**Backend service**
| Setting | Value |
|---|---|
| Root Directory | `/` (project root) |
| Dockerfile Path | `backend/Dockerfile` |
| Environment variables | `API_KEY`, `CLIENT_CODE`, `MPIN`, `TOTP_SECRET`, `FLASK_SECRET_KEY` |

After deploy, copy the backend's public URL (e.g. `https://sentimentstocks-backend.up.railway.app`).

**Frontend service**
| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Dockerfile Path | `Dockerfile` |
| Build argument | `REACT_APP_API_URL=https://sentimentstocks-backend.up.railway.app` |

The frontend will call the backend URL directly from the user's browser — no nginx proxy needed.

---

### Option 2 — Render (free tier, spins down after 15 min idle)

1. Push your repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Web Service** (repeat for each).

**Backend service**
| Setting | Value |
|---|---|
| Root Directory | `/` |
| Dockerfile Path | `./backend/Dockerfile` |
| Environment variables | `API_KEY`, `CLIENT_CODE`, `MPIN`, `TOTP_SECRET`, `FLASK_SECRET_KEY` |

**Frontend service**
| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Dockerfile Path | `./Dockerfile` |
| Build argument | `REACT_APP_API_URL=https://your-backend.onrender.com` |

---

### Option 3 — Any Linux VPS (DigitalOcean, Hetzner, Oracle Free Tier, etc.)

SSH into your server, install Docker, clone the repo, and run:

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh

# Clone repo
git clone https://github.com/your-username/SentimentStocks.git
cd SentimentStocks

# Create .env
cp .env.example .env
nano .env    # fill in credentials

# Start
docker compose up -d --build

# App is at http://<your-server-ip>
```

For HTTPS, point a domain at the server and add a Certbot/nginx reverse proxy in front.

---

## Retraining the model

If you update the data and want to retrain:

```bash
# Local
./venv/Scripts/python.exe backend/train_model.py

# Inside a running container
docker compose exec backend python backend/train_model.py
```

New model artifacts land in `models/`. Restart the backend to load them:

```bash
docker compose restart backend
```

---

## Environment variable reference

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | Angel One SmartAPI key |
| `CLIENT_CODE` | Yes | Angel One client ID |
| `MPIN` | Yes | 4-digit MPIN |
| `TOTP_SECRET` | Yes | Base32 TOTP secret (from Angel One authenticator setup) |
| `FLASK_SECRET_KEY` | Yes | Random secret for Flask sessions |

---

## Troubleshooting

**Backend won't start — `No module named 'lightgbm'`**  
The pip install step failed during build. Rebuild with `--no-cache`:
```bash
docker compose build --no-cache backend
```

**Frontend shows "No response from server"**  
The frontend can't reach the backend. Check:
- Local: both containers are running (`docker compose ps`)
- Cloud: `REACT_APP_API_URL` is set to the correct backend URL (no trailing slash)

**Angel One auth fails on first start**  
The app will perform a TOTP login automatically on first request. Check logs:
```bash
docker compose logs backend | grep -i "token\|auth\|angel"
```

**Prediction uses CSV instead of live data**  
Angel One credentials are missing or wrong. Check the `/health` endpoint — `angel_auth` should be `true`.
