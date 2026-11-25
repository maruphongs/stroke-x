"""
Stroke-X: Clinical Diagnostic Report Exporter
Generates clean, printable HTML clinical consultation reports and JSON records.
"""

import base64
import datetime
import json
from typing import Optional

import cv2
import numpy as np

from engine import StrokeAnalysisResult


def image_to_base64(img_bgr: np.ndarray, ext: str = ".png") -> str:
    """Encodes a BGR OpenCV image into a base64 data URI string."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def generate_clinical_html_report(
    result: StrokeAnalysisResult,
    case_id: str = "CASE-001",
    facility: str = "Neurovascular Emergency Center",
    clinician_notes: str = "",
) -> str:
    """
    Produces a self-contained, publication-ready, printable HTML medical report.
    Can be printed directly to PDF via standard browser print dialog.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_stroke = (result.snn_class == "STROKE")
    banner_color = "#dc2626" if is_stroke else "#16a34a"
    status_label = "CRITICAL: DEFICIT DETECTED" if is_stroke else "NORMAL: SYMMETRICAL CTA"

    overlay_b64 = image_to_base64(result.diagnostic_overlay)
    composite_b64 = image_to_base64(result.composite_dashboard)

    # Arc-profile table rows
    profile_rows_html = ""
    for row in result.arc_profile:
        lo, hi = row.get("arc_range", (0, 0))
        ai = row.get("asymmetry_index", 0.0)
        flag = '<span style="color:#dc2626; font-weight:bold;">DEFICIT</span>' if abs(ai) >= 0.35 else "Symmetric"
        profile_rows_html += f"""
        <tr>
            <td>Zone {lo:.2f} - {hi:.2f}</td>
            <td style="text-align:right;">{row.get('left_px', 0)} px</td>
            <td style="text-align:right;">{row.get('right_px', 0)} px</td>
            <td style="text-align:right; font-weight:bold;">{ai:+.3f}</td>
            <td style="text-align:center;">{flag}</td>
        </tr>
        """

    # Recommendations list
    recs_html = "".join([f"<li style='margin-bottom:6px;'>{r}</li>" for r in result.recommendations])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stroke-X Clinical Consultation Report - {case_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 30px;
            color: #1e293b;
            background-color: #ffffff;
            font-size: 13px;
            line-height: 1.5;
        }}
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 16px;
            margin-bottom: 20px;
        }}
        .title {{
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }}
        .subtitle {{
            font-size: 12px;
            color: #64748b;
            margin-top: 4px;
        }}
        .verdict-banner {{
            background-color: {banner_color};
            color: #ffffff;
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            background: #f8fafc;
        }}
        .card h4 {{
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 14px;
            color: #0f172a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th, td {{
            padding: 6px 10px;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f1f5f9;
            text-align: left;
            font-weight: 600;
            color: #475569;
        }}
        .preview-img {{
            width: 100%;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            margin-top: 8px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 12px;
            border-top: 1px solid #e2e8f0;
            font-size: 11px;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
        }}
        @media print {{
            body {{ padding: 15px; font-size: 12px; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header-bar">
        <div>
            <div class="title">STROKE-X CLINICAL TRIAGE CONSULTATION</div>
            <div class="subtitle">{facility} | Antigravity Stroke AI Suite v2.0</div>
        </div>
        <div style="text-align:right;">
            <div><strong>Case Ref:</strong> {case_id}</div>
            <div><strong>Report Date:</strong> {now_str}</div>
        </div>
    </div>

    <div class="verdict-banner">
        <div>{result.verdict}</div>
        <div style="font-size: 13px; font-weight: normal;">{status_label}</div>
    </div>

    <div class="grid-2">
        <div class="card">
            <h4>Quantitative Radiological Metrics</h4>
            <table>
                <tr><td><strong>AI SNN Stroke Probability</strong></td><td style="text-align:right;">{result.snn_prob * 100:.1f}%</td></tr>
                <tr><td><strong>SNN Classification</strong></td><td style="text-align:right; font-weight:bold;">{result.snn_class}</td></tr>
                <tr><td><strong>Identified Deficit Hemisphere</strong></td><td style="text-align:right; font-weight:bold; color:{banner_color};">{result.affected_side}</td></tr>
                <tr><td><strong>High-Deficit Voids (BBoxes)</strong></td><td style="text-align:right;">{len(result.bounding_boxes)} flagged</td></tr>
                <tr><td><strong>Scan Field of View</strong></td><td style="text-align:right;">{result.fov_type} ({result.fov_conf*100:.1f}%)</td></tr>
                <tr><td><strong>Axial Midline Tilt Angle</strong></td><td style="text-align:right;">{result.meta.get('angle', 0.0):+.2f}&deg;</td></tr>
            </table>
        </div>

        <div class="card">
            <h4>Diagnostic Deficit Overlay</h4>
            {f'<img class="preview-img" src="{overlay_b64}" alt="Diagnostic Overlay" />' if overlay_b64 else '<p>No image available</p>'}
        </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
        <h4>Anatomical Arc Radial Profile</h4>
        <table>
            <thead>
                <tr>
                    <th>Anatomical Sector</th>
                    <th style="text-align:right;">Left Hemisphere</th>
                    <th style="text-align:right;">Right Hemisphere</th>
                    <th style="text-align:right;">Asymmetry Index (AI)</th>
                    <th style="text-align:center;">Screening Status</th>
                </tr>
            </thead>
            <tbody>
                {profile_rows_html if profile_rows_html else '<tr><td colspan="5">No arc profile available</td></tr>'}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h4>Radiological Consultation Recommendations</h4>
        <ul style="padding-left: 20px; margin: 0;">
            {recs_html}
        </ul>
        {f'<div style="margin-top:12px; padding:10px; background:#fff; border-left:3px solid #0284c7;"><strong>Clinician Notes:</strong> {clinician_notes}</div>' if clinician_notes else ''}
    </div>

    <div class="footer">
        <div>Generated by Stroke-X Decision Support Pipeline | ResNet-18 SNN + MobileNetV3 FOV</div>
        <div>Confidential Medical Diagnostic Record &mdash; For Physician Decision Support Only</div>
    </div>
</body>
</html>
"""
    return html
