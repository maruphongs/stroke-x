# Early Ischemic Stroke Detection in CT Angiography Images using Image Processing

An automated neurovascular screening web dashboard for **Acute Ischemic Stroke (AIS)** and **Large Vessel Occlusion (LVO)** detection on 2D coronal/MIP Brain Computed Tomography Angiography (CTA) images.

> **Disclaimer:** This software is intended for **research, educational, and decision-support demonstration only**. It is not a certified medical device and does not replace independent diagnosis by a qualified radiologist or neurologist.

---

## Overview & Pipeline

```
[Brain CTA Image(s)]
         │
         ▼
1. FOV Classifier (MobileNetV3-Small)     ──► Identifies HEAD_ONLY vs. FULL_ARCH
         │
         ▼
2. Image Processing & Alignment           ──► Watermark removal, midline straightening,
                                              tilt compensation, and chest guillotine cut
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
3. Multi-Scale Spatial SNN (ResNet-18)     4. Morphological Vessel Deficit Mapping
   Hemispheric asymmetry scoring              Contralateral arterial gap analysis
   (SNN Score + Affected Hemisphere)          (Uniform red deficit highlight mask)
         │                                          │
         └────────────────────┬─────────────────────┘
                              ▼
5. Web Triage Dashboard & Exportable Reports (PNG, HTML/PDF, JSON, CSV)
```

---

## Dashboard Features

- **Automatic Upload (Single, Multi-File, Multi-Folder, & ZIP):** Select a folder via *Browse* (click *Browse* again to add more folders) or drag-and-drop single images, multiple images, folders, or `.zip` archives.
- **Color-Coded Triage Output:** Clear red (`ASYMMETRY DETECTED`) and green (`SYMMETRICAL SCAN`) verdict banners with **SNN Score** and **Affected Hemisphere** cards.
- **Toggleable Red Deficit Mask:** Compare the **Original CTA Scan** side-by-side with either the **Diagnostic Result (Red Deficit Mask)** or the clean **Aligned Scan**.
- **Full Batch Review:** Displays summary KPIs, a risk-sorted summary table, CSV/JSON batch downloads, and every individual scan's visual result directly on the page.
- **Report Exports:** Download result images (`PNG`), printable clinical consultation reports (`HTML / PDF`), and structured quantitative data (`JSON`).

---

## Repository Structure

```
stroke-x/
├── app.py                  # Streamlit web dashboard
├── engine.py               # Headless diagnostic pipeline & auto weight loader
├── exporter.py             # Printable HTML/PDF & JSON clinical report generator
├── preprocess.py           # Watermark stripping, midline alignment, chest cut
├── analyze.py              # Morphological vessel analysis & red deficit mask
├── fov_ai.py               # Field-of-View classifier (MobileNetV3-Small)
├── snn_ai.py               # Multi-Scale Spatial SiameseNet (ResNet-18 + BatchNorm1d)
├── .streamlit/
│   └── config.toml         # Upload limit, headless config, and light theme
├── requirements.txt        # Headless Python dependencies
├── packages.txt            # System packages for cloud builds (libgl1, libglib2.0-0)
├── Dockerfile              # Production container build
├── docker-compose.yml      # Local Docker Compose setup
├── DEPLOYMENT.md           # Step-by-step cloud & Docker deployment guide
├── .gitignore              # Excludes datasets, weights, venv, and legacy scripts
└── README.md               # Project documentation
```

---

## Quickstart (Run Locally)

### 1. Install Dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add Model Weights
Place the trained checkpoints in the project root directory (or set `SNN_MODEL_URL` and `FOV_MODEL_URL` environment variables so the app downloads them automatically):
- `best_model.pt` — Multi-Scale Spatial SiameseNet (ResNet-18)
- `fov_model.pt` — Field-of-View Classifier (MobileNetV3-Small)

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Deployment

You can deploy this dashboard for free on **Hugging Face Spaces (16 GB RAM, recommended)**, **Streamlit Community Cloud**, or locally via **Docker**:

```bash
docker compose up -d --build
```

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for full step-by-step deployment instructions for every platform.
