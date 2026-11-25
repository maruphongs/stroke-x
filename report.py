import argparse
import datetime
import json
import os

import cv2
import numpy as np


def draw_card(canvas, x, y, w, h, title="", bg_color=(255, 255, 255), border_color=(226, 232, 240), title_color=(30, 41, 59)):
    """Draws a sleek medical card panel with title header in Light Mode."""
    # Card Background
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    # Card Border
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)
    # Header bar
    if title:
        header_h = 32
        cv2.rectangle(canvas, (x, y), (x + w, y + header_h), (248, 250, 252), -1)
        cv2.line(canvas, (x, y + header_h), (x + w, y + header_h), border_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, title, (x + 12, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, title_color, 1, cv2.LINE_AA)


def draw_pill(canvas, x, y, text, bg_color, text_color=(255, 255, 255), font_scale=0.36, pad_x=10, pad_y=4):
    """Draws a rounded badge/pill with high-contrast text."""
    (tw, th), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    w = tw + pad_x * 2
    h = th + pad_y * 2
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (max(0, bg_color[0] - 25), max(0, bg_color[1] - 25), max(0, bg_color[2] - 25)), 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x + pad_x, y + pad_y + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
    return w, h


def draw_meter_bar(canvas, x, y, w, h, value, max_val=100.0, label="", color=(30, 30, 220)):
    """Draws a horizontal progress/metric bar in Light Mode."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (226, 232, 240), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (203, 213, 225), 1, cv2.LINE_AA)
    pct = max(0.0, min(1.0, float(value) / float(max_val)))
    fill_w = int(w * pct)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color, -1)
    if label:
        cv2.putText(canvas, label, (x + w + 8, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (51, 65, 85), 1, cv2.LINE_AA)


def fit_image_into_box(src_img, target_w, target_h, bg_color=(255, 255, 255)):
    """Resizes src_img keeping aspect ratio and centers it in target dimensions."""
    box = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    if src_img is None or src_img.size == 0:
        return box
    sh, sw = src_img.shape[:2]
    if sh == 0 or sw == 0:
        return box
    scale = min(target_w / sw, target_h / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(src_img, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    box[oy:oy + nh, ox:ox + nw] = resized
    return box


def generate_dashboard_image(
    image_path: str,
    final_image: np.ndarray,
    debug_dash: np.ndarray,
    triage: dict,
    verdict: str,
    sub_banner: str = "",
    findings: dict | None = None,
    meta: dict | None = None,
    output_dir: str = "reports",
    original_aligned: np.ndarray | None = None,
):
    """
    Assembles and renders an image-focused, publication-grade Light Mode medical
    diagnostic dashboard image (1920x1120) showcasing primary diagnostic scans,
    baseline reference scans, and the 6-stage algorithmic pipeline.
    """
    canvas_w = 1920
    canvas_h = 1120
    # Clean Light Mode background (soft slate off-white)
    canvas = np.full((canvas_h, canvas_w, 3), (248, 249, 250), dtype=np.uint8)

    # Diagnostic tier color scheme (Light Mode)
    verdict_upper = verdict.upper()
    if "ASYMMETRY" in verdict_upper:
        tier_color = (30, 30, 220)       # Crimson Red (BGR)
        accent_color = (40, 40, 255)
    else:
        tier_color = (40, 165, 50)       # Emerald Green (BGR)
        accent_color = (55, 195, 65)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ==========================================
    # 1. TOP HEADER BAR (y: 0 to 66)
    # ==========================================
    cv2.rectangle(canvas, (0, 0), (canvas_w, 66), (255, 255, 255), -1)
    cv2.line(canvas, (0, 0), (canvas_w, 0), accent_color, 3)
    cv2.line(canvas, (0, 66), (canvas_w, 66), (226, 232, 240), 1)

    # Scan & FOV metadata
    meta = meta or {}
    fov_name = meta.get("fov_type") or "CTA SCAN"
    tilt_ang = meta.get("angle", 0.0)
    chest_cut = meta.get("chest_y")
    cut_txt = f"Guillotine y={chest_cut}" if chest_cut else "Head Only"
    meta_txt = f"Case: {os.path.basename(image_path)}  |  FOV: {fov_name}  |  Tilt: {tilt_ang:+.2f} deg  |  {cut_txt}"
    cv2.putText(canvas, meta_txt, (720, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (51, 65, 85), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Generated: {now_str}  |  Radiological Vision Mode (Light)", (720, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

    # Top-right Verdict Badge
    (vtw, _vth), _ = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    badge_w = vtw + 26
    badge_x = canvas_w - badge_w - 20
    badge_y = 12
    badge_h = 42
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), tier_color, -1)
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), accent_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, verdict, (badge_x + 13, badge_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # ==========================================
    # 2. MAIN IMAGE VISUAL SHOWCASE (y: 74 to 1038)
    # ==========================================
    aff_side = findings.get("affected_side", "NONE") if findings else "NONE"
    n_boxes = findings.get("bbox_count", 0) if findings else 0

    # --- Left Column: Primary Scans (x: 16 to 626, w: 610) ---
    # Card A: Primary Diagnostic Scan (Deficit Overlays + Midline + BBoxes)
    draw_card(canvas, 16, 74, 610, 544, title="1. PRIMARY CTA DIAGNOSTIC MAP (DEFICIT & BBOXES)")
    draw_pill(canvas, 500, 78, "DIAGNOSTIC", (200, 100, 0), font_scale=0.30, pad_x=6, pad_y=2)
    fitted_cta = fit_image_into_box(final_image, 586, 464)
    canvas[112:112 + 464, 28:28 + 586] = fitted_cta
    cv2.rectangle(canvas, (28, 112), (28 + 586, 112 + 464), (226, 232, 240), 1)
    # Caption bar under image
    cv2.rectangle(canvas, (17, 584), (16 + 609, 617), (248, 250, 252), -1)
    cv2.line(canvas, (16, 584), (16 + 610, 584), (226, 232, 240), 1)
    cap_a = f"Deficit: {aff_side}  |  Flagged High-Deficit Regions: {n_boxes} (>=85% threshold)"
    cv2.putText(canvas, cap_a, (28, 604), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (51, 65, 85), 1, cv2.LINE_AA)

    # Card B: Aligned Baseline CTA Reference Scan
    ref_scan = original_aligned if (original_aligned is not None and original_aligned.size > 0) else meta.get("raw_clean", final_image)
    draw_card(canvas, 16, 626, 610, 412, title="2. ALIGNED BASELINE CTA SCAN (REFERENCE)")
    draw_pill(canvas, 514, 630, "BASELINE", (71, 85, 105), font_scale=0.30, pad_x=6, pad_y=2)
    fitted_base = fit_image_into_box(ref_scan, 586, 334)
    canvas[664:664 + 334, 28:28 + 586] = fitted_base
    cv2.rectangle(canvas, (28, 664), (28 + 586, 664 + 334), (226, 232, 240), 1)
    # Caption bar under image
    cv2.rectangle(canvas, (17, 1004), (16 + 609, 1037), (248, 250, 252), -1)
    cv2.line(canvas, (16, 1004), (16 + 610, 1004), (226, 232, 240), 1)
    cap_b = f"Intracranial Alignment: {tilt_ang:+.2f} deg tilt compensation  |  Skull-Stripped"
    cv2.putText(canvas, cap_b, (28, 1024), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (51, 65, 85), 1, cv2.LINE_AA)

    # --- Right Section: 6-Stage Scientific Architecture Pipeline (x: 638 to 1904, w: 1266, h: 964) ---
    draw_card(canvas, 638, 74, 1266, 964, title="3. SCIENTIFIC ARCHITECTURE & 6-STAGE PIPELINE ANALYSIS")
    draw_pill(canvas, 1740, 78, "6-STAGE PIPELINE", (2, 132, 199), font_scale=0.30, pad_x=6, pad_y=2)

    if debug_dash is not None and debug_dash.size > 0:
        _dh, dw = debug_dash.shape[:2]
        stored_panels = getattr(debug_dash, "panels", None)
        if stored_panels and len(stored_panels) == 6:
            panels_to_draw = stored_panels
        elif dw >= 5 * _dh:
            pw = dw // 6
            panels_to_draw = [debug_dash[:, i * pw:(i + 1) * pw] for i in range(6)]
        else:
            pw = dw // 4
            panels_to_draw = [debug_dash[:, i * pw:(i + 1) * pw] for i in range(min(6, 4))]

        # 2 rows x 3 columns grid of prominent large panels
        col_w = 402
        row_h = 442
        start_px = 654
        start_py = 118
        gap_x = 16
        gap_y = 14

        for i in range(min(6, len(panels_to_draw))):
            r = i // 3
            c = i % 3
            px = start_px + c * (col_w + gap_x)
            py = start_py + r * (row_h + gap_y)

            # Prominent Image Box taking up the full cell (394 x 434 px)
            img_w = col_w - 8
            img_h = row_h - 8
            img_x = px + 4
            img_y = py + 4

            # Container card for pipeline cell
            cv2.rectangle(canvas, (px, py), (px + col_w, py + row_h), (255, 255, 255), -1)
            cv2.rectangle(canvas, (px, py), (px + col_w, py + row_h), (226, 232, 240), 1, cv2.LINE_AA)

            fitted_p = fit_image_into_box(panels_to_draw[i], img_w, img_h)
            canvas[img_y:img_y + img_h, img_x:img_x + img_w] = fitted_p
            cv2.rectangle(canvas, (img_x, img_y), (img_x + img_w, img_y + img_h), (226, 232, 240), 1)

    # ==========================================
    # 3. BOTTOM SLEEK METRIC STRIP (y: 1046 to 1114, Height: 68px)
    # ==========================================
    cv2.rectangle(canvas, (16, 1046), (canvas_w - 16, 1114), (255, 255, 255), -1)
    cv2.rectangle(canvas, (16, 1046), (canvas_w - 16, 1114), (226, 232, 240), 1, cv2.LINE_AA)

    # SNN Score Progress Meter
    snn_prob = findings.get("snn_prob") if findings else None
    cv2.putText(canvas, "AI Siamese Network (SNN) Asymmetry Probability:", (32, 1085),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (15, 23, 42), 1, cv2.LINE_AA)
    if snn_prob is not None:
        snn_cls = "STROKE" if snn_prob >= 0.50 else "HEALTHY"
        draw_meter_bar(canvas, 370, 1071, 220, 18, snn_prob * 100, max_val=100.0,
                       label=f"{snn_prob * 100:.1f}% ({snn_cls})",
                       color=(30, 30, 220) if snn_prob >= 0.50 else (40, 165, 50))
    else:
        cv2.putText(canvas, "Morphology Only", (370, 1085), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA)

    # Status Pills
    pill_x = 680
    pill_w, _ = draw_pill(canvas, pill_x, 1069, f"DEFICIT: {aff_side}", (200, 100, 0) if aff_side != "NONE" else (71, 85, 105), font_scale=0.36)
    pill_x += pill_w + 12

    pill_w, _ = draw_pill(canvas, pill_x, 1069, f"FLAGGED REGIONS: {n_boxes}", (30, 30, 220) if n_boxes > 0 else (40, 165, 50), font_scale=0.36)
    pill_x += pill_w + 12

    pill_w, _ = draw_pill(canvas, pill_x, 1069, f"TILT: {tilt_ang:+.2f} DEG", (71, 85, 105), font_scale=0.36)
    pill_x += pill_w + 12

    draw_pill(canvas, pill_x, 1069, f"CASE: {os.path.basename(image_path)}", (100, 116, 139), font_scale=0.36)

    # Footer note on bottom right
    cv2.putText(canvas, "Antigravity Stroke Suite v2.0  |  Radiological Vision Mode",
                (canvas_w - 410, 1085), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (148, 163, 184), 1, cv2.LINE_AA)

    # Save to reports/
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{stem}_dashboard.png"
    out_path = os.path.join(output_dir, out_filename)
    try:
        cv2.imwrite(out_path, canvas)
        print(f"[+] Final dashboard image saved: {out_path}")
    except Exception as e:
        print(f"[!] Note: Could not write dashboard image ({e})")

    return canvas, out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Comprehensive Final Stroke Dashboard Image or View Reports")
    parser.add_argument("--image", type=str, default=None, help="Scan image to generate dashboard for")
    parser.add_argument("--report", type=str, default=None, help="Path to existing JSON report to rebuild dashboard")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory (default: reports)")
    parser.add_argument("--windows", action="store_true", help="Display analysis in separate windows with CLI text report")
    parser.add_argument("--no-gui", action="store_true", help="Do not open GUI windows (useful for headless / batch / CLI text only)")

    args = parser.parse_args()
    mode = "windows" if args.windows else "dashboard"

    if args.report:
        if args.windows:
            from report_windowed import print_cli_report, show_windowed_report
            with open(args.report, "r") as f:
                data = json.load(f)
            img_path = data.get("image_path")
            verdict = data.get("verdict", "UNKNOWN")
            sub_banner = data.get("sub_banner", "")
            findings = data.get("radiological", {})
            triage = data.get("clinical_survey", {})
            print_cli_report(
                image_path=img_path or args.report,
                verdict=verdict,
                sub_banner=sub_banner,
                findings=findings,
                triage=triage,
            )
            if not args.no_gui and img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    show_windowed_report(final_image=img, image_name=os.path.basename(img_path))
            return

        with open(args.report, "r") as f:
            data = json.load(f)
        img_path = data.get("image_path")
        if not img_path or not os.path.exists(img_path):
            print(f"[!] Image not found: {img_path}")
            return
        print(f"Rebuilding dashboard from report: {args.report} for {img_path}")
        from interface import run_inference
        run_inference(img_path, skip_survey=True, show_gui=not args.no_gui, report_mode=mode)
    elif args.image:
        from interface import run_inference
        run_inference(args.image, skip_survey=True, show_gui=not args.no_gui, report_mode=mode)
    else:
        print("Please provide --image <path> or --report <path>")


if __name__ == "__main__":
    main()
