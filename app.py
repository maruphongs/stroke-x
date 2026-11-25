"""
Stroke-X: Brain CTA Stroke Triage Dashboard
Clean, color-coded clinical UI supporting single and full-output batch upload.
"""

import json
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from engine import StrokeAnalysisResult, run_stroke_pipeline
from exporter import generate_clinical_html_report

# ------------------------------------------------------------
# 1. Page Configuration & Color-Coded Medical Styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Stroke-X | Brain CTA Triage",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .app-banner {
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .app-banner h1 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    .app-banner p {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }
    .verdict-box-stroke {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
    .verdict-box-normal {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
    .verdict-title-stroke {
        color: #991b1b;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-title-normal {
        color: #166534;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-sub {
        color: #475569;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .metric-card.red {
        border-top-color: #dc2626;
        background-color: #fffafa;
    }
    .metric-card.green {
        border-top-color: #16a34a;
        background-color: #f9fefb;
    }
    .metric-card.slate {
        border-top-color: #475569;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-value.red { color: #b91c1c; }
    .metric-value.green { color: #15803d; }
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .pill-red {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .pill-green {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #4ade80;
    }
    .case-divider {
        border-top: 2px solid #e2e8f0;
        margin: 24px 0 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. Helpers & Cached Inference
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze_cached(
    image_bytes: bytes,
    case_name: str,
    asym_threshold: float,
    bbox_threshold: float,
) -> StrokeAnalysisResult:
    return run_stroke_pipeline(
        image_input=image_bytes,
        case_name=case_name,
        asym_threshold=asym_threshold,
        bbox_threshold=bbox_threshold,
        generate_composite=False,
    )


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def render_metric_card(col, label: str, value: str, accent: str = ""):
    card_cls = f"metric-card {accent}".strip()
    val_cls = f"metric-value {accent}".strip()
    col.markdown(
        f"""
        <div class="{card_cls}">
            <div class="metric-label">{label}</div>
            <div class="{val_cls}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_result(
    case_name: str,
    result: StrokeAnalysisResult,
    show_mask: bool = True,
    key_prefix: str = "single",
    expanded_recs: bool = True,
):
    """Renders a clean, color-coded diagnostic view for one scan."""
    if not result.success:
        st.error(f"Could not analyze {case_name}: {result.error}")
        return

    is_stroke = result.snn_class == "STROKE"
    accent = "red" if is_stroke else "green"

    # 1. Color-coded Verdict Banner
    if is_stroke:
        side_txt = (
            f"{result.affected_side} Hemisphere"
            if result.affected_side in ("LEFT", "RIGHT")
            else result.affected_side
        )
        st.markdown(
            f"""
            <div class="verdict-box-stroke">
                <div class="verdict-title-stroke">
                    <span class="pill pill-red">ASYMMETRY DETECTED</span> &nbsp; {side_txt}
                </div>
                <div class="verdict-sub">Case: <b>{case_name}</b> &bull; Possible acute ischemic stroke / vessel occlusion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-box-normal">
                <div class="verdict-title-normal">
                    <span class="pill pill-green">SYMMETRICAL SCAN</span> &nbsp; No Significant Deficit
                </div>
                <div class="verdict-sub">Case: <b>{case_name}</b> &bull; Bilateral intracranial vessels appear balanced</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Color-coded Key Metrics (SNN Score & Affected Hemisphere)
    c1, c2 = st.columns(2)
    render_metric_card(c1, "SNN Score", f"{result.snn_prob * 100:.1f}%", accent)
    render_metric_card(c2, "Affected Hemisphere", result.affected_side, accent)

    # 3. Side-by-Side Visual Comparison (Original vs Aligned / Diagnostic Mask)
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original CTA Scan**")
        if result.original_image is not None:
            st.image(bgr_to_rgb(result.original_image), use_container_width=True)
    with img_col2:
        if show_mask:
            st.markdown("**Diagnostic Result (Red Deficit Mask)**")
            display_img = result.diagnostic_overlay
        else:
            st.markdown("**Aligned Scan (Mask Off)**")
            display_img = result.aligned_image if result.aligned_image is not None else result.diagnostic_overlay

        if display_img is not None:
            st.image(bgr_to_rgb(display_img), use_container_width=True)

    # 4. Clinical Guidance & Exports
    with st.expander(f"Clinical Guidance & Export — {case_name}", expanded=expanded_recs):
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

        st.markdown("---")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            export_img = result.diagnostic_overlay if show_mask else (
                result.aligned_image if result.aligned_image is not None else result.diagnostic_overlay
            )
            if export_img is not None:
                _, enc_img = cv2.imencode(".png", export_img)
                st.download_button(
                    "Download Result Image (PNG)",
                    data=enc_img.tobytes(),
                    file_name=f"{case_name}_result.png",
                    mime="image/png",
                    key=f"{key_prefix}_png",
                    use_container_width=True,
                )
        with dl2:
            html_report = generate_clinical_html_report(result=result, case_id=case_name)
            st.download_button(
                "Printable Report (HTML / PDF)",
                data=html_report,
                file_name=f"{case_name}_report.html",
                mime="text/html",
                key=f"{key_prefix}_html",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "Download Data (JSON)",
                data=json.dumps(result.to_dict(), indent=2),
                file_name=f"{case_name}_report.json",
                mime="application/json",
                key=f"{key_prefix}_json",
                use_container_width=True,
            )


# ------------------------------------------------------------
# 3. Sidebar: Upload (Single or Batch) & View Controls
# ------------------------------------------------------------
st.sidebar.markdown("### Stroke-X Controls")
st.sidebar.caption("Upload one scan for single review, or multiple scans for batch triage.")

uploaded_files = st.sidebar.file_uploader(
    "Upload CTA Scan(s)",
    type=["jpeg", "jpg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")
show_mask = st.sidebar.toggle(
    "Show Red Deficit Mask",
    value=True,
    help="Toggle between Diagnostic Result (with red mask) and clean Aligned Scan.",
)

with st.sidebar.expander("Advanced Settings", expanded=False):
    asym_threshold = st.slider(
        "SNN Cutoff (%)",
        min_value=20.0,
        max_value=80.0,
        value=50.0,
        step=1.0,
    )
    bbox_threshold = st.slider(
        "Deficit Threshold (%)",
        min_value=50.0,
        max_value=95.0,
        value=85.0,
        step=1.0,
    )


# ------------------------------------------------------------
# 4. Main Area: Banner, Queue & Results
# ------------------------------------------------------------
st.markdown(
    """
    <div class="app-banner">
        <h1>Stroke-X Brain CTA Triage</h1>
        <p>Automated Acute Ischemic Stroke &amp; Large Vessel Occlusion Screening</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not uploaded_files:
    st.info("Upload one or more CTA scan images in the sidebar to begin analysis.")
    st.stop()

scan_queue = [(uf.name, uf.getvalue()) for uf in uploaded_files]
total_scans = len(scan_queue)

# --- SINGLE MODE ---
if total_scans == 1:
    case_name, img_bytes = scan_queue[0]
    with st.spinner(f"Analyzing {case_name}..."):
        res = analyze_cached(img_bytes, case_name, asym_threshold, bbox_threshold)
    render_single_result(case_name, res, show_mask=show_mask, key_prefix="single_0", expanded_recs=True)

# --- BATCH MODE (SHOWS ALL OUTPUTS) ---
else:
    progress_bar = st.progress(0, text="Starting batch analysis...")
    t_batch_start = time.time()
    results_list = []

    for idx, (case_name, img_bytes) in enumerate(scan_queue):
        progress_bar.progress(
            idx / total_scans,
            text=f"Analyzing scan {idx + 1} of {total_scans}: {case_name}...",
        )
        res = analyze_cached(img_bytes, case_name, asym_threshold, bbox_threshold)
        results_list.append((case_name, res))

    progress_bar.progress(1.0, text=f"Completed {total_scans} scans")
    total_elapsed = time.time() - t_batch_start

    # Batch Summary Metrics
    flagged_count = sum(1 for _, r in results_list if r.success and r.snn_class == "STROKE")
    normal_count = sum(1 for _, r in results_list if r.success and r.snn_class == "HEALTHY")
    failed_count = sum(1 for _, r in results_list if not r.success)

    st.markdown("### Batch Triage Summary")
    b1, b2, b3, b4 = st.columns(4)
    render_metric_card(b1, "Total Scans", str(total_scans), "")
    render_metric_card(b2, "Flagged (Stroke Risk)", str(flagged_count), "red" if flagged_count > 0 else "")
    render_metric_card(b3, "Normal (Symmetrical)", str(normal_count), "green" if normal_count > 0 else "")
    render_metric_card(b4, "Total Time", f"{total_elapsed:.1f}s", "slate")

    if failed_count > 0:
        st.warning(f"{failed_count} scan(s) could not be processed.")

    # Build Sorted Summary Table (Flagged cases first, highest SNN score at top)
    sorted_items = []
    for case_name, r in results_list:
        if r.success:
            priority = 0 if r.snn_class == "STROKE" else 1
            status_str = "STROKE RISK" if r.snn_class == "STROKE" else "NORMAL"
            row = {
                "Scan": case_name,
                "Status": status_str,
                "Affected Side": r.affected_side,
                "SNN Score (%)": round(r.snn_prob * 100, 1),
                "Time (s)": round(r.execution_time_sec, 2),
            }
            sorted_items.append((priority, -(r.snn_prob * 100), case_name, r, row))
        else:
            row = {
                "Scan": case_name,
                "Status": "ERROR",
                "Affected Side": "-",
                "SNN Score (%)": 0.0,
                "Time (s)": round(r.execution_time_sec, 2),
            }
            sorted_items.append((2, 0.0, case_name, r, row))

    sorted_items.sort(key=lambda x: (x[0], x[1], x[2]))
    df_display = pd.DataFrame([item[4] for item in sorted_items])

    def style_status(val):
        if val == "STROKE RISK":
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        if val == "NORMAL":
            return "background-color: #dcfce7; color: #166534; font-weight: 700;"
        return "background-color: #f1f5f9; color: #475569;"

    st.dataframe(
        df_display.style.map(style_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    # Batch Export Buttons
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "Download Batch Summary (CSV)",
            data=df_display.to_csv(index=False),
            file_name="stroke_x_batch_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        batch_json = json.dumps(
            [{"scan": name, **r.to_dict()} for _, _, name, r, _ in sorted_items], indent=2
        )
        st.download_button(
            "Download All Results (JSON)",
            data=batch_json,
            file_name="stroke_x_batch_results.json",
            mime="application/json",
            use_container_width=True,
        )

    # Render ALL Batch Outputs directly
    st.markdown("### All Scan Results")
    for idx, (_, _, case_name, res, _) in enumerate(sorted_items):
        if idx > 0:
            st.markdown('<div class="case-divider"></div>', unsafe_allow_html=True)
        render_single_result(
            case_name,
            res,
            show_mask=show_mask,
            key_prefix=f"batch_{idx}",
            expanded_recs=False,
        )
