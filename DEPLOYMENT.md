# Deployment Guide

Because trained PyTorch checkpoints (`best_model.pt` ~44 MB and `fov_model.pt` ~6 MB) are excluded from the Git repository to keep it lightweight and code-only, every deployment method below includes how to supply the two `.pt` files.

---

## Option 1: Hugging Face Spaces (Recommended — Free 16 GB RAM)

Hugging Face Spaces offers **2 vCPU + 16 GB RAM for free**, which handles PyTorch + ResNet-18 + batch image processing smoothly (Streamlit Community Cloud's free tier is limited to ~1 GB RAM, which can be tight for PyTorch).

### Steps
1. Sign in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Configure the Space:
   - **Space Name:** `stroke-x`
   - **Space SDK:** **Docker** (uses your `Dockerfile` directly) or **Streamlit**
   - **Hardware:** Free (`2 vCPU • 16 GB RAM`)
3. Push this repository to your Space (or link via GitHub Actions):
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/stroke-x
   git push hf main
   ```
4. Upload the two model weights to the root of your Space:
   - Open your Space on Hugging Face → **Files** → **Add file** → **Upload files**
   - Upload `best_model.pt` and `fov_model.pt` (Hugging Face automatically handles them via Git LFS).
   - *Alternative:* Set `SNN_MODEL_URL` and `FOV_MODEL_URL` under **Settings → Variables and secrets** (using GitHub Release download URLs) and the app will auto-download them on boot.
5. Wait ~2–3 minutes for the container build. Your public dashboard URL will be:
   `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/stroke-x`

---

## Option 2: Streamlit Community Cloud (Free, Direct GitHub Sync)

> **Note:** Streamlit Community Cloud free instances have ~1–2.7 GB RAM. PyTorch CPU works for single and small batches, though large batches may hit memory limits compared to Hugging Face Spaces.

### Steps
1. Host your weights on a **GitHub Release** (free up to 2 GB per file):
   - Go to `https://github.com/maruphongs/stroke-x/releases` → **Draft a new release** (e.g., tag `v1.0.0`).
   - Attach `best_model.pt` and `fov_model.pt` as release assets and click **Publish release**.
   - Copy the direct download links for both files:
     - `https://github.com/maruphongs/stroke-x/releases/download/v1.0.0/best_model.pt`
     - `https://github.com/maruphongs/stroke-x/releases/download/v1.0.0/fov_model.pt`
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**:
   - **Repository:** `maruphongs/stroke-x`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings → Secrets** and add:
   ```toml
   SNN_MODEL_URL = "https://github.com/maruphongs/stroke-x/releases/download/v1.0.0/best_model.pt"
   FOV_MODEL_URL = "https://github.com/maruphongs/stroke-x/releases/download/v1.0.0/fov_model.pt"
   ```
5. Click **Deploy**. On first startup, `engine.py` downloads and caches both checkpoints automatically.

---

## Option 3: Local / On-Premise Docker (Private Network)

Best when patient scans must stay strictly on a local workstation or hospital intranet.

### Prerequisites
- Docker Desktop / Docker Engine installed
- `best_model.pt` and `fov_model.pt` placed in the project root folder

### Commands
```bash
# Build and start in background
docker compose up -d --build

# Check logs
docker compose logs -f

# Stop
docker compose down
```
Open `http://localhost:8501` in your browser.

---

## Option 4: Cloud Containers (Render / Railway / Google Cloud Run)

1. Connect `https://github.com/maruphongs/stroke-x` as a Docker service (auto-detects `Dockerfile`).
2. Expose port `8501` and allocate at least **2–4 GB RAM**.
3. Add environment variables `SNN_MODEL_URL` and `FOV_MODEL_URL` pointing to your GitHub Release assets.
