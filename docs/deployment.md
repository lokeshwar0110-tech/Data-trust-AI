# DataTrust AI — Deployment Guide

This guide details options for deploying the DataTrust AI Platform across development, local daemon, containerized, and production cloud environments.

---

## Architecture Overview

DataTrust AI runs as an asynchronous, high-throughput service:
- **Core Engine:** Mathematical quality scoring, Analytic Hierarchy Process (AHP), Defect Policies, and Downstream Validation.
- **REST API:** FastAPI application exposing endpoints for task auto-detection, evaluation, calibration, remediation simulation, and cryptographic audit receipts.
- **Web Dashboard:** Single-page application served via static mount at `/` with interactive radar charts, task selectors, and PDF export.

---

## 1. Local & Daemon Execution

### Active Local Daemon
The platform is pre-configured with a direct launch script:

```bash
# From the project root
python run.py
```

Or run via Uvicorn directly:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 2. Containerized Deployment (Docker & Docker Compose)

### Using Docker Compose (Recommended for Local/On-Premise)

```bash
# Build and launch in detached mode
docker compose up -d --build

# View container logs
docker compose logs -f

# Check health status
docker compose ps

# Shutdown
docker compose down
```

### Using Plain Docker

```bash
# Build the Docker image
docker build -t datatrust-ai:latest .

# Run the container
docker run -d \
  --name datatrust-ai \
  -p 8000:8000 \
  --restart unless-stopped \
  datatrust-ai:latest

# Verify health check
curl -f http://localhost:8000/health
```

---

## 3. Production Cloud Deployment

### Google Cloud Run (Serverless Container)
```bash
# 1. Build and push image to Google Artifact Registry
gcloud builds submit --tag gcr.io/[PROJECT_ID]/datatrust-ai:latest

# 2. Deploy to Cloud Run
gcloud run deploy datatrust-ai \
  --image gcr.io/[PROJECT_ID]/datatrust-ai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 1
```

### AWS App Runner / ECS Fargate
1. Push image to Amazon ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com
   docker tag datatrust-ai:latest [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/datatrust-ai:latest
   docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/datatrust-ai:latest
   ```
2. In AWS App Runner or ECS console, configure:
   - **Port:** `8000`
   - **Health Check Path:** `/health`
   - **Protocol:** HTTP

### Render / Fly.io
Deploy directly using the repository's `Dockerfile`.
- **Fly.io:**
  ```bash
  fly launch
  fly deploy
  ```

---

## 4. Systemd Service (Linux Server / VM)

Create `/etc/systemd/system/datatrust.service`:

```ini
[Unit]
Description=DataTrust AI Platform Service
After=network.target

[Service]
Type=simple
User=datatrust
Group=datatrust
WorkingDirectory=/opt/datatrust-ai
ExecStart=/opt/datatrust-ai/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
Environment="PATH=/opt/datatrust-ai/venv/bin"
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable datatrust
sudo systemctl start datatrust
sudo systemctl status datatrust
```

---

## 5. Reverse Proxy Configuration (Nginx)

```nginx
server {
    listen 80;
    server_name datatrust.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (for live logs/updates)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```
