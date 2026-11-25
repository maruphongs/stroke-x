# Stroke-X: Neurovascular Acute Ischemic Stroke & LVO AI Rapid Triage Suite

Stroke-X is an automated neurovascular decision-support platform designed to rapidly detect **Acute Ischemic Stroke (AIS)** and **Large Vessel Occlusion (LVO)** from Brain Computed Tomography Angiography (CTA) scans.

---

## 🧠 Diagnostic Architecture

Stroke-X combines deep learning and computational morphology to emulate expert neuroradiology assessment:

```
[Brain CTA Scan]
       │
       ▼
1. FOV AI Classifier (MobileNetV3-Small)  ──► Detects HEAD_ONLY vs. FULL_ARCH
       │
       ▼
2. Morphological Preprocessing            ──► Watermark scrubbing, midline tilt compensation,
                                              and aortic arch guillotine cut
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
3. Hemispheric Siamese Network (ResNet-18)  4. Morphological Vascular Arc Profile
   Contralateral contrastive comparison       Intracranial arterial arborization &
   (93.8% Benchmark Cohort Accuracy)           regional deficit void bounding boxes
       │                                         │
       └──────────────────┬──────────────────────┘
                          ▼
5. Integrated Diagnostic Verdict & Clinical Consultation Report
```

---

## 🖥️ Interactive Web Dashboard

Launch the clinician diagnostic portal:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit dashboard
streamlit run app.py
```

### Dashboard Capabilities:
* **One-Click Cohort Ingestion:** Test healthy (`n1`–`n10`) and stroke (`s1`–`s10`) scans directly, or upload arbitrary CTA slices.
* **1920x1120 Diagnostic Infographic:** Displays the full 6-stage algorithmic progression.
* **Side-by-Side Multi-View:** Ingested Scan &rarr; Straightened Midline &rarr; Deficit Heatmap Overlay.
* **Dynamic Sensitivity Sliders:** Adjust Asymmetry Probability (default 50%) and Deficit Area Bounding Box (default 85%) thresholds in real time.
* **Clinical FAST Checklist:** Correlates focal motor/speech deficits with neuroimaging asymmetry.
* **Multi-Format Export:** Export structured findings in JSON, PNG diagnostic board, or printable HTML/PDF reports.

---

## 📁 Repository Structure

```
stroke-x/
├── app.py                     # Streamlit interactive web diagnostic dashboard
├── engine.py                  # Headless core inference service & data structures
├── exporter.py                # Printable HTML and JSON clinical report generator
├── analyze.py                 # Morphological spatial analysis & arc-length profile
├── evaluate.py                # Model validation metrics & confusion matrix generator
├── fov_ai.py                  # Field-of-View classification network (MobileNetV3)
├── interface.py               # Unified CLI inference coordinator
├── preprocess.py              # Text stripping, midline alignment, skull stripping
├── report.py                  # 1920x1120 composite infographic dashboard renderer
├── report_windowed.py         # Multi-window and CLI structured text reporting
├── snn_ai.py                  # Siamese Neural Network architecture and training
├── survey.py                  # Clinical symptom (FAST/BE-FAST) triage module
├── test.py                    # Automated cohort test suite runner
├── run_snn_comparison.py      # Entry point for SNN architecture comparisons
├── snn_comparison/            # Model comparison scripts and benchmarks
├── Dockerfile                 # Production container image definition
├── docker-compose.yml         # Container orchestration configuration
├── DEPLOYMENT.md              # Deployment guide (Hugging Face Spaces, Docker, Cloud)
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Strict data & environment exclusion
└── README.md                  # Project documentation
```

---

## 🐳 Docker Deployment

Run the entire suite locally with Docker in one command:

```bash
docker compose up -d --build
```
Open your browser at `http://localhost:8501`.

For cloud deployment instructions (Hugging Face Spaces, Render, Railway, Google Cloud Run), refer to [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 🧪 Automated Testing

Execute the automated batch test runner over test cohorts:

```bash
# Headless batch test across cohort scans
python test.py --data-dir data/test --report-mode windows
```

---

## 📄 License & Disclaimer
This software is intended for research and clinical decision support demonstration. It does not replace independent evaluation by a certified neuroradiologist or neurologist.
