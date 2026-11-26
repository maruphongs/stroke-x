"""
Brain CTA Stroke Triage Dashboard
Clean, color-coded clinical UI supporting single, multi-file, multi-folder, and ZIP upload.
"""

import io
import json
import os
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from engine import StrokeAnalysisResult, ensure_model_weights, run_stroke_pipeline
from exporter import generate_clinical_html_report

# ------------------------------------------------------------
# 1. Page Configuration & Color-Coded Medical Styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Early Ischemic Stroke Detection in CT Angiography Images using Image Processing",
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo("assets/icon.png", icon_image="assets/icon.png", size="large")

st.markdown(
    """
    <style>
    .app-banner {
        background-color: #111827;
        color: #f9fafb;
        padding: 20px 26px;
        border-radius: 4px;
        border: 1px solid #1f2937;
        border-left: 5px solid #14b8a6;
        margin-bottom: 18px;
    }
    .app-banner-eyebrow {
        display: inline-block;
        background-color: #1f2937;
        color: #5eead4;
        border: 1px solid #374151;
        border-radius: 3px;
        padding: 2px 9px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .app-banner h1 {
        color: #f9fafb;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.25;
    }
    .app-banner p {
        color: #9ca3af;
        font-size: 0.94rem;
        margin: 6px 0 0 0;
    }
    .verdict-box-stroke {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 14px;
    }
    .verdict-box-normal {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #15803d;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 14px;
    }
    .verdict-title-stroke {
        color: #991b1b;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-title-normal {
        color: #166534;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-sub {
        color: #4b5563;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-top: 3px solid #0f766e;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-card.red {
        border-top-color: #dc2626;
        background-color: #fffafa;
    }
    .metric-card.green {
        border-top-color: #15803d;
        background-color: #f9fefb;
    }
    .metric-card.slate {
        border-top-color: #374151;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-value.red { color: #b91c1c; }
    .metric-value.green { color: #15803d; }
    .pill {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 3px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .pill-red {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .pill-green {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .step-badge {
        display: inline-block;
        background-color: #f0fdfa;
        color: #0f766e;
        border: 1px solid #99f6e4;
        border-radius: 3px;
        padding: 2px 9px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .case-divider {
        border-top: 1px solid #d1d5db;
        margin: 24px 0 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. Helpers, Model Setup & Cached Inference
# ------------------------------------------------------------
def _get_secret(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except Exception:
        return None


@st.cache_resource(show_spinner="Checking AI model checkpoints...")
def init_model_weights() -> tuple[bool, bool]:
    return ensure_model_weights(
        snn_url=_get_secret("SNN_MODEL_URL"),
        fov_url=_get_secret("FOV_MODEL_URL"),
    )


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


def render_how_to_use():
    """Renders a simple 4-step quick-start guide and status legend."""
    steps = [
        (
            "Step 1",
            ":material/upload_file: Upload CTA scans",
            "Use the left sidebar to select a folder (click **Browse** again to add more folders), or drag and drop single images, multiple files, or a **.zip** archive.",
        ),
        (
            "Step 2",
            ":material/neurology: Automatic analysis",
            "Each image is automatically straightened along the brain midline, scored by the Siamese Neural Network (**SNN Score**), and evaluated for vessel asymmetry.",
        ),
        (
            "Step 3",
            ":material/compare: Review & toggle mask",
            "Compare the **Original CTA Scan** against the **Diagnostic Result**. Use **Show Red Deficit Mask** in the sidebar to switch between masked and clean aligned views.",
        ),
        (
            "Step 4",
            ":material/download: Export reports",
            "Download individual result images (**PNG**), printable clinical reports (**HTML / PDF**), structured **JSON**, or a full batch summary (**CSV** / **JSON**).",
        ),
    ]

    cols = st.columns(4)
    for col, (badge, title, desc) in zip(cols, steps):
        with col.container(border=True, height="stretch"):
            st.markdown(f'<span class="step-badge">{badge}</span>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown(
        """
        <div style="margin-top: 8px; padding: 10px 14px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 4px; font-size: 0.88rem; color: #374151;">
            <b>Result guide:</b> &nbsp;
            <span class="pill pill-red">ASYMMETRY DETECTED</span> SNN Score &ge; cutoff (default 50%) &mdash; possible ischemic stroke / vessel occlusion. &nbsp;&nbsp;
            <span class="pill pill-green">SYMMETRICAL SCAN</span> SNN Score &lt; cutoff &mdash; balanced bilateral vessels.
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
        st.error(f"Could not analyze {case_name}: {result.error}", title="Scan processing error")
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
            st.image(
                bgr_to_rgb(result.original_image),
                alt=f"Original CTA scan for {case_name}",
                width="stretch",
            )
    with img_col2:
        if show_mask:
            st.markdown("**Diagnostic Result (Red Deficit Mask)**")
            display_img = result.diagnostic_overlay
            alt_text = f"Aligned CTA scan with red deficit highlight mask for {case_name}"
        else:
            st.markdown("**Aligned Scan (Mask Off)**")
            display_img = result.aligned_image if result.aligned_image is not None else result.diagnostic_overlay
            alt_text = f"Midline-aligned CTA scan without overlay mask for {case_name}"

        if display_img is not None:
            st.image(bgr_to_rgb(display_img), alt=alt_text, width="stretch")

    # 4. Clinical Guidance & Exports
    with st.expander(f"Clinical guidance & export — {case_name}", expanded=expanded_recs):
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

        st.divider()
        safe_file_id = os.path.basename(case_name)
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            export_img = result.diagnostic_overlay if show_mask else (
                result.aligned_image if result.aligned_image is not None else result.diagnostic_overlay
            )
            if export_img is not None:
                _, enc_img = cv2.imencode(".png", export_img)
                st.download_button(
                    "Download result image (PNG)",
                    data=enc_img.tobytes(),
                    file_name=f"{safe_file_id}_result.png",
                    mime="image/png",
                    key=f"{key_prefix}_png",
                    icon=":material/image:",
                    width="stretch",
                )
        with dl2:
            html_report = generate_clinical_html_report(result=result, case_id=case_name)
            st.download_button(
                "Printable report (HTML / PDF)",
                data=html_report,
                file_name=f"{safe_file_id}_report.html",
                mime="text/html",
                key=f"{key_prefix}_html",
                icon=":material/description:",
                width="stretch",
            )
        with dl3:
            st.download_button(
                "Download data (JSON)",
                data=json.dumps(result.to_dict(), indent=2),
                file_name=f"{safe_file_id}_report.json",
                mime="application/json",
                key=f"{key_prefix}_json",
                icon=":material/data_object:",
                width="stretch",
            )


def extract_scans_from_uploads(uploaded_items) -> list[tuple[str, bytes]]:
    """Extracts image scans from uploaded files, directories, and ZIP archives."""
    valid_exts = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
    scans = []
    seen = set()

    for uf in uploaded_items:
        name = uf.name
        raw = uf.getvalue()
        ext = os.path.splitext(name)[1].lower()

        if ext == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for member in sorted(zf.namelist()):
                        base = os.path.basename(member)
                        if not base or base.startswith(".") or "__MACOSX" in member:
                            continue
                        if os.path.splitext(base)[1].lower() in valid_exts:
                            data = zf.read(member)
                            sig = (member, len(data))
                            if sig not in seen:
                                seen.add(sig)
                                scans.append((member, data))
            except zipfile.BadZipFile:
                st.warning(f"Skipped invalid ZIP archive: {name}", title="Invalid archive")
        elif ext in valid_exts:
            base = os.path.basename(name)
            if not base.startswith("."):
                sig = (name, len(raw))
                if sig not in seen:
                    seen.add(sig)
                    scans.append((name, raw))

    return scans


# ------------------------------------------------------------
# 3. Sidebar: Upload (Folders, Files, or ZIP - Automatic)
# ------------------------------------------------------------
snn_ready, fov_ready = init_model_weights()

with st.sidebar:
    st.markdown("### Controls")
    st.caption("Drop files, ZIPs, or folders below (click Browse to select a folder).")

    uploaded_files = st.file_uploader(
        "Upload CTA scan(s) or folder(s)",
        type=["jpeg", "jpg", "png", "bmp", "tiff", "tif", "zip"],
        accept_multiple_files="directory",
        key="unified_uploader",
        help="Select a folder via Browse (click Browse again to add more folders), or drag and drop files, folders, or ZIPs.",
    )

    show_mask = st.toggle(
        "Show red deficit mask",
        value=True,
        key="show_mask_toggle",
        help="Toggle between the Diagnostic Result (red highlight) and the clean Aligned Scan.",
    )

    with st.expander("Advanced settings", expanded=False, icon=":material/tune:"):
        asym_threshold = st.slider(
            "SNN cutoff (%)",
            min_value=20.0,
            max_value=80.0,
            value=50.0,
            step=1.0,
            key="snn_cutoff_slider",
        )
        bbox_threshold = st.slider(
            "Deficit threshold (%)",
            min_value=50.0,
            max_value=95.0,
            value=85.0,
            step=1.0,
            key="bbox_thresh_slider",
        )

    st.space("small")
    st.caption("Model status")
    with st.container(horizontal=True):
        st.badge(
            "SNN ready" if snn_ready else "SNN missing",
            icon=":material/check_circle:" if snn_ready else ":material/error:",
            color="green" if snn_ready else "red",
        )
        st.badge(
            "FOV ready" if fov_ready else "FOV missing",
            icon=":material/check_circle:" if fov_ready else ":material/error:",
            color="green" if fov_ready else "red",
        )


# ------------------------------------------------------------
# 4. Main Area: Banner, Notices, How-to-Use & Results
# ------------------------------------------------------------
st.markdown(
    """
    <div class="app-banner">
        <div class="app-banner-eyebrow">Neurovascular CTA AI Screening</div>
        <h1>Early Ischemic Stroke Detection in CT Angiography Images using Image Processing</h1>
        <p>Automated Acute Ischemic Stroke &amp; Large Vessel Occlusion Screening</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Startup Notices (always visible at top)
st.warning(
    "This application is for **research, educational, and clinical decision-support demonstration only**. "
    "It is not a certified medical device and does not replace evaluation by a qualified radiologist or neurologist. "
    "Upload **2D coronal/MIP brain CTA images** only (JPEG, PNG, TIFF, BMP, or ZIP) and ensure patient-identifying information is anonymized before uploading.",
    title="Important notice",
    icon=":material/warning:",
)

if not snn_ready or not fov_ready:
    missing = []
    if not snn_ready:
        missing.append("`best_model.pt` (SNN classifier)")
    if not fov_ready:
        missing.append("`fov_model.pt` (FOV classifier)")
    st.error(
        f"Missing model checkpoint(s): {', '.join(missing)}. "
        "Place them in the project root directory or configure `SNN_MODEL_URL` and `FOV_MODEL_URL` in your deployment secrets/environment variables.",
        title="Model weights not loaded",
        icon=":material/error:",
    )

if not uploaded_files:
    st.subheader("How to use")
    render_how_to_use()
else:
    with st.expander("How to use", expanded=False, icon=":material/help:"):
        render_how_to_use()

    scan_queue = extract_scans_from_uploads(uploaded_files)
    total_scans = len(scan_queue)

    if total_scans == 0:
        st.warning(
            "No supported CTA scan images (JPEG, PNG, BMP, TIFF) were found in the uploaded selection.",
            title="No valid images found",
        )
    elif total_scans == 1:
        case_name, img_bytes = scan_queue[0]
        with st.spinner(f"Analyzing {case_name}..."):
            res = analyze_cached(img_bytes, case_name, asym_threshold, bbox_threshold)
        render_single_result(case_name, res, show_mask=show_mask, key_prefix="single_0", expanded_recs=True)
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

        st.subheader("Batch triage summary")
        b1, b2, b3, b4 = st.columns(4)
        render_metric_card(b1, "Total Scans", str(total_scans), "")
        render_metric_card(b2, "Flagged (Stroke Risk)", str(flagged_count), "red" if flagged_count > 0 else "")
        render_metric_card(b3, "Normal (Symmetrical)", str(normal_count), "green" if normal_count > 0 else "")
        render_metric_card(b4, "Total Time", f"{total_elapsed:.1f}s", "slate")

        if failed_count > 0:
            st.warning(f"{failed_count} scan(s) could not be processed.", title="Partial batch errors")

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
            width="stretch",
            hide_index=True,
            alt="Batch triage summary table showing scan names, stroke risk status, affected hemisphere, SNN score, and compute time",
        )

        # Batch Export Buttons
        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "Download batch summary (CSV)",
                data=df_display.to_csv(index=False),
                file_name="stroke_x_batch_summary.csv",
                mime="text/csv",
                icon=":material/table_view:",
                width="stretch",
            )
        with exp2:
            batch_json = json.dumps(
                [{"scan": name, **r.to_dict()} for _, _, name, r, _ in sorted_items], indent=2
            )
            st.download_button(
                "Download all results (JSON)",
                data=batch_json,
                file_name="stroke_x_batch_results.json",
                mime="application/json",
                icon=":material/data_object:",
                width="stretch",
            )

        # Render ALL Batch Outputs directly
        st.subheader("All scan results")
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
