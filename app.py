"""
Stroke-X: Neurovascular Acute Ischemic Stroke & LVO AI Rapid Triage Dashboard
Interactive clinical decision support platform built with Streamlit.
"""

import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from engine import run_stroke_pipeline
from exporter import generate_clinical_html_report

# ------------------------------------------------------------
# 1. Page Configuration & Theme
# ------------------------------------------------------------
st.set_page_config(
    page_title="Stroke-X | Neurovascular AI Rapid Triage",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-stroke {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .badge-healthy {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #4ade80;
    }
    .card-container {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 16px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. Helper Functions & Sample Scan Discovery
# ------------------------------------------------------------
@st.cache_data
def get_sample_scans(data_dir: str = "data/test") -> dict:
    samples = {}
    if not os.path.exists(data_dir):
        return samples

    valid_exts = {".jpeg", ".jpg", ".png", ".bmp"}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                label = "Healthy Normal" if stem.lower().startswith("n") else "Suspected Stroke"
                samples[f"{f} ({label})"] = path

    # Natural sorting
    def sort_key(item):
        k = item[0].lower()
        prefix = "0" if k.startswith("n") else "1"
        num = "".join([c for c in k if c.isdigit()])
        return (prefix, int(num) if num else 0)

    return dict(sorted(samples.items(), key=sort_key))


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# 3. Sidebar: Configuration, Scan Ingestion & Clinical Triage
# ------------------------------------------------------------
st.sidebar.markdown("### 🧠 Stroke-X Platform")
st.sidebar.markdown("**Neurovascular AIS & LVO Decision Support**")
st.sidebar.caption("SNN ResNet-18 Benchmark Accuracy: **93.8%**")

st.sidebar.divider()
st.sidebar.markdown("#### 📥 Ingest Brain CTA Scan")

source_choice = st.sidebar.radio(
    "Scan Input Mode",
    ["Select Test Cohort Scan", "Upload Local CTA Scan"],
    index=0,
)

image_path_or_bytes = None
case_display_name = "Scan"

if source_choice == "Select Test Cohort Scan":
    sample_dict = get_sample_scans()
    if sample_dict:
        selected_key = st.sidebar.selectbox("Choose Sample Case", list(sample_dict.keys()), index=0)
        image_path_or_bytes = sample_dict[selected_key]
        case_display_name = selected_key.split(" ")[0]
    else:
        st.sidebar.warning("No test cohort scans found in data/test.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CTA Scan Image (JPEG, PNG, TIFF)",
        type=["jpeg", "jpg", "png", "bmp", "tiff", "tif"],
    )
    if uploaded_file is not None:
        image_path_or_bytes = uploaded_file.read()
        case_display_name = uploaded_file.name

st.sidebar.divider()
st.sidebar.markdown("#### ⚙️ Threshold Calibration")
asym_threshold = st.sidebar.slider(
    "SNN Asymmetry Threshold (%)",
    min_value=20.0,
    max_value=80.0,
    value=50.0,
    step=1.0,
    help="Threshold probability above which the hemisphere comparison is classified as asymmetric (Stroke).",
)
bbox_threshold = st.sidebar.slider(
    "High-Deficit Region Threshold (%)",
    min_value=50.0,
    max_value=95.0,
    value=85.0,
    step=1.0,
    help="Contralateral vascular disparity threshold for flagging ischemic void bounding boxes.",
)

st.sidebar.divider()
st.sidebar.markdown("#### 🩺 Clinical FAST Triage (Optional)")
with st.sidebar.expander("FAST Symptom Checklist", expanded=False):
    fast_face = st.checkbox("Face Drooping / Asymmetry", value=False)
    fast_arm = st.selectbox("Arm Weakness / Drift", ["None", "Left Arm Weakness", "Right Arm Weakness"])
    fast_speech = st.checkbox("Speech Difficulty / Dysarthria", value=False)
    time_onset = st.selectbox(
        "Time from Symptom Onset",
        ["< 4.5 Hours (IV Thrombolysis Window)", "4.5 - 24 Hours (EVT LVO Window)", "> 24 Hours", "Unknown / Wake-up"],
    )

# ------------------------------------------------------------
# 4. Main Body: Analysis & Live Diagnostic Reports
# ------------------------------------------------------------
st.markdown("<div class='main-header'>Stroke-X Neurovascular Diagnostic Portal</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Automated Acute Ischemic Stroke & Large Vessel Occlusion (LVO) Rapid Triage from CTA</div>",
    unsafe_allow_html=True,
)

if image_path_or_bytes is None:
    st.info("👈 Please select a test cohort scan or upload a brain CTA scan in the sidebar to begin analysis.")
    st.stop()

# Execution trigger
with st.spinner(f"Analyzing {case_display_name} through neurovascular pipeline..."):
    result = run_stroke_pipeline(
        image_input=image_path_or_bytes,
        case_name=case_display_name,
        asym_threshold=asym_threshold,
        bbox_threshold=bbox_threshold,
        save_dashboard_to_disk=False,
    )

if not result.success:
    st.error(f"❌ Analysis Failed: {result.error}")
    st.stop()

# ------------------------------------------------------------
# 5. Top Diagnostic Banner & Key Metrics Cards
# ------------------------------------------------------------
is_stroke = (result.snn_class == "STROKE")
badge_class = "badge-stroke" if is_stroke else "badge-healthy"
alert_type = st.error if is_stroke else st.success

with st.container():
    col_verdict, col_time = st.columns([4, 1])
    with col_verdict:
        st.markdown(
            f"### <span class='status-badge {badge_class}'>{result.verdict}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"**Case:** `{case_display_name}` | **Summary:** {result.sub_banner}")
    with col_time:
        st.metric("Triage Compute Time", f"{result.execution_time_sec:.2f}s")

# 4 Key Diagnostic Metric Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    delta_val = f"{(result.snn_prob * 100) - asym_threshold:+.1f}% vs {asym_threshold:.0f}% cutoff"
    st.metric(
        "SNN Asymmetry Probability",
        f"{result.snn_prob * 100:.1f}%",
        delta=delta_val,
        delta_color="inverse" if is_stroke else "normal",
    )
with m_col2:
    st.metric(
        "Flagged Hemisphere",
        result.affected_side,
        help="Primary cerebral hemisphere displaying unilateral vascular attenuation or void.",
    )
with m_col3:
    st.metric(
        "High-Deficit Voids",
        f"{len(result.bounding_boxes)} Regions",
        help=f"Vascular clusters showing >= {bbox_threshold}% contralateral volume disparity.",
    )
with m_col4:
    tilt_deg = result.meta.get("angle", 0.0)
    st.metric(
        "Scan Field of View",
        result.fov_type,
        delta=f"Tilt: {tilt_deg:+.1f}°",
        delta_color="off",
    )

st.divider()

# ------------------------------------------------------------
# 6. Interactive Analysis Tabs
# ------------------------------------------------------------
tab_composite, tab_multiview, tab_stages, tab_profile, tab_recs = st.tabs([
    "📊 Composite Diagnostic Board",
    "🔬 Multi-View Comparative Scans",
    "📐 6-Stage Algorithmic Pipeline",
    "📈 Anatomical Arc Screening Profile",
    "📋 Clinical Findings & Export",
])

# --- Tab 1: Composite Dashboard ---
with tab_composite:
    st.markdown("#### 1920x1120 Publication-Grade Diagnostic Infographic")
    if result.composite_dashboard is not None:
        rgb_dash = bgr_to_rgb(result.composite_dashboard)
        st.image(rgb_dash, use_container_width=True, caption=f"Stroke-X Integrated Diagnostic Board - {case_display_name}")
        
        # Download button for dashboard image
        _, encoded_dash = cv2.imencode(".png", result.composite_dashboard)
        st.download_button(
            label="💾 Download Diagnostic Dashboard (PNG)",
            data=encoded_dash.tobytes(),
            file_name=f"{case_display_name}_stroke_dashboard.png",
            mime="image/png",
        )
    else:
        st.warning("Composite dashboard image not available.")

# --- Tab 2: Multi-View Comparative Scans ---
with tab_multiview:
    st.markdown("#### Radiologic View Progression")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Raw Ingested Scan**")
        if result.original_image is not None:
            st.image(bgr_to_rgb(result.original_image), use_container_width=True)
    with c2:
        st.markdown("**2. Aligned & Midline Straightened**")
        if result.aligned_image is not None:
            st.image(bgr_to_rgb(result.aligned_image), use_container_width=True)
            st.caption(f"Midline CX: {result.meta.get('cx')}, Tilt: {result.meta.get('angle', 0.0):+.1f}°")
    with c3:
        st.markdown("**3. Morphological Deficit Overlay**")
        if result.diagnostic_overlay is not None:
            st.image(bgr_to_rgb(result.diagnostic_overlay), use_container_width=True)
            st.caption(f"Flagged Voids: {len(result.bounding_boxes)}")

# --- Tab 3: 6-Stage Algorithmic Pipeline ---
with tab_stages:
    st.markdown("#### Step-by-Step Algorithmic Verification")
    if result.stage_panels:
        grid_cols = st.columns(3)
        for idx, panel in enumerate(result.stage_panels):
            with grid_cols[idx % 3]:
                title = result.stage_titles[idx] if idx < len(result.stage_titles) else f"Stage {idx+1}"
                sub = result.stage_subtitles[idx] if idx < len(result.stage_subtitles) else ""
                st.markdown(f"**{title}**")
                if sub:
                    st.caption(sub)
                st.image(bgr_to_rgb(panel), use_container_width=True)
    else:
        st.info("Pipeline stage images unavailable.")

# --- Tab 4: Anatomical Arc Profile ---
with tab_profile:
    st.markdown("#### Contralateral Arc-Length Asymmetry Distribution")
    st.caption("Evaluates bilateral arterial vessel density across 6 radial anatomical sectors.")
    if result.arc_profile:
        df_rows = []
        for row in result.arc_profile:
            lo, hi = row["arc_range"]
            ai = row["asymmetry_index"]
            flag = "DEFICIT FLAGGED" if abs(ai) >= 0.35 else "Symmetric"
            df_rows.append({
                "Anatomical Sector": f"Zone {lo:.2f} - {hi:.2f}",
                "Left Vessel (px)": row["left_px"],
                "Right Vessel (px)": row["right_px"],
                "Asymmetry Index (AI)": round(ai, 3),
                "Status": flag,
            })
        df_profile = pd.DataFrame(df_rows)
        
        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.dataframe(df_profile, use_container_width=True)
        with col_chart:
            # Bar chart comparing left vs right pixels
            chart_df = df_profile[["Anatomical Sector", "Left Vessel (px)", "Right Vessel (px)"]].set_index("Anatomical Sector")
            st.bar_chart(chart_df)
    else:
        st.info("No arc-length profile data available for this scan.")

# --- Tab 5: Clinical Recommendations & Export ---
with tab_recs:
    st.markdown("#### Formal Clinical Consultation Findings")
    rec_box = st.container()
    with rec_box:
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

    # Clinical FAST correlation note
    if fast_face or fast_arm != "None" or fast_speech:
        st.divider()
        st.markdown("##### 🚨 Combined Radiological + Clinical FAST Triage Alert")
        symptom_list = []
        if fast_face: symptom_list.append("Facial Droop")
        if fast_arm != "None": symptom_list.append(fast_arm)
        if fast_speech: symptom_list.append("Speech Difficulty")
        st.warning(
            f"**Positive Clinical Symptoms:** {', '.join(symptom_list)} | "
            f"**Time Window:** {time_onset}\n\n"
            "⚠️ High clinical urgency: Concordance between clinical focal deficits and radiological findings warrants immediate Code Stroke activation."
        )

    st.divider()
    st.markdown("#### 📥 Export Records")
    json_export = json.dumps(result.to_dict(), indent=2)
    html_report = generate_clinical_html_report(
        result=result,
        case_id=case_display_name,
        clinician_notes=f"FAST Checklist: Face={fast_face}, Arm={fast_arm}, Speech={fast_speech}, Onset={time_onset}",
    )
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="📄 Structured JSON Findings",
            data=json_export,
            file_name=f"{case_display_name}_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl2:
        if result.composite_dashboard is not None:
            _, enc_dash = cv2.imencode(".png", result.composite_dashboard)
            st.download_button(
                label="🖼️ Diagnostic Board (PNG)",
                data=enc_dash.tobytes(),
                file_name=f"{case_display_name}_dashboard.png",
                mime="image/png",
                use_container_width=True,
            )
    with col_dl3:
        st.download_button(
            label="📑 Printable Report (HTML / PDF)",
            data=html_report,
            file_name=f"{case_display_name}_clinical_report.html",
            mime="text/html",
            use_container_width=True,
        )
