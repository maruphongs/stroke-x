# Stroke-X: Deployment Guide

This guide details three straightforward deployment pathways for the **Stroke-X** Neurovascular AI diagnostic platform:

1. **Hugging Face Spaces (Free Cloud Demo)** &mdash; Instant public web link with 16 GB RAM.
2. **Docker / Docker Compose** &mdash; On-premise / Local hospital private deployment.
3. **Cloud PaaS (Railway / Render / Google Cloud Run)** &mdash; Production container hosting.

---

## 🚀 Option 1: Hugging Face Spaces (Recommended for Fast Public Demo)

Hugging Face Spaces provides a free, hosted Streamlit runtime with **16 GB RAM and 2 vCPUs**, ideal for medical computer vision models.

### Step-by-Step Instructions:

1. Log in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Set Space properties:
   - **Space Name:** `stroke-x`
   - **License:** MIT or Apache 2.0
   - **Space SDK:** **Streamlit**
   - **Space Hardware:** Free (2 vCPU &bull; 16 GB RAM)
3. Connect your GitHub repository:
   - In Space Settings &rarr; **Git Repository / Sync with GitHub**, select `https://github.com/maruphongs/stroke-x`.
4. Upload Model Checkpoints:
   - Because weights (`best_model.pt` ~44MB and `fov_model.pt` ~6MB) are excluded from the main code repo, upload them to the root of your Hugging Face Space using the web interface or Git LFS:
     ```bash
     git clone https://huggingface.co/spaces/YOUR_USERNAME/stroke-x
     cp path/to/best_model.pt stroke-x/
     cp path/to/fov_model.pt stroke-x/
     cd stroke-x
     git add best_model.pt fov_model.pt
     git commit -m "add model checkpoints"
     git push
     ```
5. Your dashboard will build automatically and be live at `https://huggingface.co/spaces/YOUR_USERNAME/stroke-x`!

---

## 🐳 Option 2: Local / On-Premise Docker Deployment

For clinical environments where patient brain scans must never leave the hospital network (HIPAA / PDPA / GDPR compliance):

### Prerequisites:
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed.
- `best_model.pt` and `fov_model.pt` located in the project directory.

### Quick Launch:
```bash
# 1. Build and start the container in detached mode
docker compose up -d --build

# 2. View logs
docker compose logs -f

# 3. Access the dashboard
# Open your browser at http://localhost:8501
```

### Stop Service:
```bash
docker compose down
```

---

## ☁️ Option 3: Cloud Container Deployment (Render / Railway / GCP)

### Railway / Render:
1. Create a new Web Service and link your GitHub repository `https://github.com/maruphongs/stroke-x`.
2. Select **Docker** as the deployment runtime (it will automatically detect `Dockerfile`).
3. Set the exposed port to `8501`.
4. Upload or mount model weights as persistent volume or download them via release asset.

### Google Cloud Run (Serverless):
```bash
# Build with Google Cloud Build
gcloud builds submit --tag gcr.io/PROJECT_ID/stroke-x

# Deploy to Cloud Run with 4GB memory
gcloud run deploy stroke-x \
    --image gcr.io/PROJECT_ID/stroke-x \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --port 8501 \
    --allow-unauthenticated
```
