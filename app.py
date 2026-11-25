"""
Stroke-X: Simple & Fast Brain CTA Stroke Triage Dashboard
Supports both single-scan analysis and multi-scan batch triage.
"""

import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from engine import StrokeAnalysisResult, run_stroke_pipeline
from exporter import generate_clinical_html_report

# ------------------------------------------------------------
# 1. Page Configuration & Clean Styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Stroke-X | Brain CTA Triage",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .verdict-box-stroke {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .verdict-box-normal {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .verdict-title-stroke {
        color: #991b1b;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-title-normal {
        color: #166534;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-sub {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------
@st.cache_data
def get_sample_scans(data_dir: str = "data/test") -> dict:
    samples = {}
    if not os.path.exists(data_dir):
        return samples

    valid_exts = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                tag = "Normal" if stem.lower().startswith("n") else "Stroke"
                samples[f"{f} ({tag})"] = path

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


def render_single_result(case_name: str, result: StrokeAnalysisResult):
    """Renders a clean, easy-to-understand diagnostic view for one scan."""
    if not result.success:
        st.error(f"❌ Could not analyze **{case_name}**: {result.error}")
        return

    is_stroke = result.snn_class == "STROKE"

    # 1. Clear Verdict Banner
    if is_stroke:
        side_txt = (
            f"{result.affected_side} Hemisphere"
            if result.affected_side in ("LEFT", "RIGHT")
            else result.affected_side
        )
        st.markdown(
            f"""
            <div class="verdict-box-stroke">
                <div class="verdict-title-stroke">⚠️ Asymmetry Detected — {side_txt}</div>
                <div class="verdict-sub">Case: <b>{case_name}</b> &bull; Possible acute ischemic stroke / vessel occlusion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-box-normal">
                <div class="verdict-title-normal">✅ Symmetrical Scan — No Significant Deficit</div>
                <div class="verdict-sub">Case: <b>{case_name}</b> &bull; Bilateral intracranial vessels appear balanced</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Three Simple Key Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("AI Stroke Risk Score", f"{result.snn_prob * 100:.1f}%")
    c2.metric("Affected Hemisphere", result.affected_side)
    c3.metric("Highlighted Deficit Areas", f"{len(result.bounding_boxes)}")

    # 3. Side-by-Side Image Comparison (Original vs AI Result)
    st.markdown("#### Visual Comparison")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original CTA Scan**")
        if result.original_image is not None:
            st.image(bgr_to_rgb(result.original_image), use_container_width=True)
    with img_col2:
        st.markdown("**AI Result (Highlighted Deficit Areas)**")
        if result.diagnostic_overlay is not None:
            st.image(bgr_to_rgb(result.diagnostic_overlay), use_container_width=True)

    # 4. Clinical Next Steps & Export
    with st.expander("📋 Clinical Guidance & Export Report", expanded=True):
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

        st.markdown("---")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            if result.diagnostic_overlay is not None:
                _, enc_img = cv2.imencode(".png", result.diagnostic_overlay)
                st.download_button(
                    "🖼️ Download Result Image (PNG)",
                    data=enc_img.tobytes(),
                    file_name=f"{case_name}_result.png",
                    mime="image/png",
                    use_container_width=True,
                )
        with dl2:
            html_report = generate_clinical_html_report(result=result, case_id=case_name)
            st.download_button(
                "📑 Printable Report (HTML / PDF)",
                data=html_report,
                file_name=f"{case_name}_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "📄 Download Data (JSON)",
                data=json.dumps(result.to_dict(), indent=2),
                file_name=f"{case_name}_report.json",
                mime="application/json",
                use_container_width=True,
            )


# ------------------------------------------------------------
# 3. Sidebar: Upload (Single or Batch) & Optional Settings
# ------------------------------------------------------------
st.sidebar.title("🧠 Stroke-X")
st.sidebar.caption("Fast Brain CTA Stroke Triage")

uploaded_files = st.sidebar.file_uploader(
    "Upload CTA Scan(s)",
    type=["jpeg", "jpg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
    help="Upload 1 image for single analysis, or multiple images for batch triage.",
)

sample_dict = get_sample_scans()
selected_samples = []
if sample_dict and not uploaded_files:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Or Try Sample Scans**")
    run_all_samples = st.sidebar.checkbox("Select all sample scans (Batch demo)", value=False)
    if run_all_samples:
        selected_samples = list(sample_dict.keys())
        st.sidebar.info(f"Selected all {len(selected_samples)} sample scans.")
    else:
        default_sample = [list(sample_dict.keys())[0]]
        selected_samples = st.sidebar.multiselect(
            "Choose 1 or more samples",
            options=list(sample_dict.keys()),
            default=default_sample,
        )

with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    asym_threshold = st.slider(
        "Stroke Sensitivity Cutoff (%)",
        min_value=20.0,
        max_value=80.0,
        value=50.0,
        step=1.0,
    )
    bbox_threshold = st.slider(
        "Deficit Highlight Threshold (%)",
        min_value=50.0,
        max_value=95.0,
        value=85.0,
        step=1.0,
    )


# ------------------------------------------------------------
# 4. Main Area: Build Queue & Run Analysis
# ------------------------------------------------------------
st.title("🧠 Stroke-X Brain CTA Triage")
st.caption("Upload a single CTA scan or a batch of scans to screen for acute stroke asymmetry.")

# Build list of (case_name, image_input)
scan_queue = []
if uploaded_files:
    for uf in uploaded_files:
        scan_queue.append((uf.name, uf.getvalue()))
elif selected_samples:
    for key in selected_samples:
        case_name = key.split(" ")[0]
        scan_queue.append((case_name, sample_dict[key]))

if not scan_queue:
    st.info("👈 Upload one or more CTA scan images in the sidebar (or choose sample scans) to start.")
    st.stop()

# Run inference (single or batch)
results_list = []
total_scans = len(scan_queue)

if total_scans == 1:
    case_name, img_data = scan_queue[0]
    with st.spinner(f"Analyzing {case_name}..."):
        res = run_stroke_pipeline(
            image_input=img_data,
            case_name=case_name,
            asym_threshold=asym_threshold,
            bbox_threshold=bbox_threshold,
            generate_composite=False,
        )
    render_single_result(case_name, res)

else:
    # Batch Mode
    progress_bar = st.progress(0, text="Starting batch analysis...")
    t_batch_start = time.time()

    for idx, (case_name, img_data) in enumerate(scan_queue):
        progress_bar.progress(
            idx / total_scans,
            text=f"Analyzing scan {idx + 1} of {total_scans}: {case_name}...",
        )
        res = run_stroke_pipeline(
            image_input=img_data,
            case_name=case_name,
            asym_threshold=asym_threshold,
            bbox_threshold=bbox_threshold,
            generate_composite=False,
        )
        results_list.append((case_name, res))

    progress_bar.progress(1.0, text=f"Completed {total_scans} scans!")
    total_elapsed = time.time() - t_batch_start

    # Batch Summary Metrics
    flagged_count = sum(1 for _, r in results_list if r.success and r.snn_class == "STROKE")
    normal_count = sum(1 for _, r in results_list if r.success and r.snn_class == "HEALTHY")
    failed_count = sum(1 for _, r in results_list if not r.success)

    st.markdown("### 📊 Batch Triage Summary")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total Scans", total_scans)
    b2.metric("Flagged (Stroke Risk)", flagged_count)
    b3.metric("Normal (Symmetrical)", normal_count)
    b4.metric("Total Time", f"{total_elapsed:.1f}s")

    if failed_count > 0:
        st.warning(f"{failed_count} scan(s) could not be processed.")

    # Build Summary Table (sorted: Flagged cases first, highest risk at top)
    table_rows = []
    for case_name, r in results_list:
        if r.success:
            status_str = "⚠️ STROKE RISK" if r.snn_class == "STROKE" else "✅ NORMAL"
            table_rows.append(
                {
                    "Scan": case_name,
                    "Status": status_str,
                    "Affected Side": r.affected_side,
                    "Stroke Risk (%)": round(r.snn_prob * 100, 1),
                    "Deficit Areas": len(r.bounding_boxes),
                    "Time (s)": round(r.execution_time_sec, 2),
                    "_sort_priority": 0 if r.snn_class == "STROKE" else 1,
                }
            )
        else:
            table_rows.append(
                {
                    "Scan": case_name,
                    "Status": "❌ ERROR",
                    "Affected Side": "-",
                    "Stroke Risk (%)": 0.0,
                    "Deficit Areas": 0,
                    "Time (s)": round(r.execution_time_sec, 2),
                    "_sort_priority": 2,
                }
            )

    df_batch = pd.DataFrame(table_rows).sort_values(
        by=["_sort_priority", "Stroke Risk (%)"], ascending=[True, False]
    )
    df_display = df_batch.drop(columns=["_sort_priority"])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Batch Export Buttons
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "📊 Download Batch Summary (CSV)",
            data=df_display.to_csv(index=False),
            file_name="stroke_x_batch_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        batch_json = json.dumps(
            [{"scan": name, **r.to_dict()} for name, r in results_list], indent=2
        )
        st.download_button(
            "📄 Download All Results (JSON)",
            data=batch_json,
            file_name="stroke_x_batch_results.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # Interactive Case Inspector for Batch Results
    st.markdown("### 🔍 Inspect Individual Scan")
    ordered_names = df_display["Scan"].tolist()
    result_map = dict(results_list)
    chosen_case = st.selectbox("Select a scan from the batch to view details", ordered_names)
    if chosen_case:
        render_single_result(chosen_case, result_map[chosen_case])
