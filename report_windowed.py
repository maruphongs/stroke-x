import argparse
import datetime
import json
import os

import cv2
import numpy as np


# ============================================================
# 1. COMMAND-LINE STRUCTURED TEXT REPORT
# ============================================================
def print_cli_report(
    image_path: str,
    verdict: str,
    sub_banner: str = "",
    findings: dict | None = None,
    triage: dict | None = None,
    meta: dict | None = None,
    debug_dash: np.ndarray | None = None
):
    """
    Prints a clean, comprehensive clinical diagnostic report directly to the terminal/command line.
    """
    findings = findings or {}
    triage = triage or {}
    meta = meta or {}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stem = os.path.basename(image_path)
    fov_name = meta.get("fov_type") or "CTA SCAN"
    tilt_ang = meta.get("angle", 0.0)
    chest_cut = meta.get("chest_y")
    cut_txt = f"Guillotine y={chest_cut}" if chest_cut else "No chest cut"

    snn_prob = findings.get("snn_prob") if findings.get("snn_prob") is not None else findings.get("snn_probability")
    snn_str = f"{snn_prob * 100:.1f}%" if snn_prob is not None else "N/A"
    aff_side = findings.get("affected_side") or findings.get("deficit_side", "NONE")
    n_boxes = findings.get("bbox_count", 0)

    # 1. Header
    print("\n" + "=" * 80)
    print("           NEUROVASCULAR STROKE AI RAPID TRIAGE REPORT (CLI MODE)")
    print("=" * 80)
    print(f" Case: {stem}  |  FOV: {fov_name}  |  Tilt: {tilt_ang:+.2f} deg  |  {cut_txt}")
    print(f" Generated: {now_str}  |  Antigravity Stroke Suite v2.0")
    print("-" * 80)

    # 2. Integrated Verdict
    print(" [INTEGRATED DIAGNOSTIC VERDICT]")
    print(f"  >>> {verdict} <<<")
    if sub_banner:
        print(f"  Details: {sub_banner}")
    print("-" * 80)

    # 3. Quantitative Radiological Metrics
    snn_cls = "STROKE" if (snn_prob is not None and snn_prob >= 0.50) else "HEALTHY"
    print(" [SECTION 1: QUANTITATIVE CTA RADIOLOGICAL METRICS]")
    print(f"  • AI Siamese Network (SNN) Class : {snn_cls} (Asymmetry Prob: {snn_str} | Benchmark Acc: 93.8%)")
    print(f"  • Detected Deficit Hemisphere   : {aff_side}")
    boxes = getattr(debug_dash, "bounding_boxes", []) or findings.get("bounding_boxes", [])
    if boxes:
        print(f"  • Flagged High-Deficit BBoxes   : {len(boxes)} region(s) (>= 85.0% deficit threshold)")
        for i, b in enumerate(boxes, 1):
            print(f"     -> Box #{i}: Area {b['area']:.0f}px | Asym {b['asymmetry'] if 'asymmetry' in b else b.get('asym', 0):.1f}% at (x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']})")

    # 4. Anatomical Arc-Length Profile
    profile = getattr(debug_dash, "profile", None)
    if profile:
        print("\n [SECTION 2: ANATOMICAL ARC SCREENING PROFILE]")
        print("   Anatomical Zone     Left (px)   Right (px)    AI      Screening Flag")
        print("   -------------------------------------------------------------------")
        for row in profile:
            lo, hi = row["arc_range"]
            ai = row["asymmetry_index"]
            flag = " <-- SIGNIFICANT DEFICIT" if abs(ai) >= 0.35 else ""
            side = " (L>R)" if ai < -0.35 else (" (R>L)" if ai > 0.35 else "")
            print(f"   Zone {lo:0.2f} - {hi:0.2f}        {row['left_px']:6d}      {row['right_px']:6d}   {ai:+.3f}  {flag}{side}")

    # 5. Radiological Recommendations
    verdict_upper = verdict.upper()
    print("\n [SECTION 3: RADIOLOGICAL FINDINGS & RECOMMENDATIONS]")
    if "ASYMMETRY" in verdict_upper:
        recs = [
            f"1. Significant unilateral vascular asymmetry flagged in the {aff_side} hemisphere.",
            f"2. {n_boxes} high-deficit region(s) detected with >=85% contralateral vessel disparity.",
            "3. Prompt clinical evaluation and correlation with patient focal motor deficits advised.",
            "4. Consider urgent advanced perfusion imaging (CTP / MRI DWI) to assess salvageable penumbra."
        ]
    else:
        recs = [
            "1. CTA scan demonstrates bilateral symmetrical intracranial arterial arborization.",
            "2. Negative for proximal large vessel occlusion (ICA / M1 / Basilar asymmetry).",
            "3. If clinical deficits persist, evaluate for lacunar, posterior fossa, or small vessel disease.",
            "4. Consider MRI Brain with DWI for non-LVO acute ischemic infarction if clinically indicated."
        ]

    for line in recs:
        print(f"   {line}")

    print("=" * 80 + "\n")


# ============================================================
# 2. IMAGE EXPORT & MULTI-WINDOW DISPLAY FUNCTIONS
# ============================================================
def save_windowed_images(
    final_image: np.ndarray,
    original_aligned: np.ndarray | None = None,
    debug_dash: np.ndarray | None = None,
    image_name: str = "Scan",
    output_dir: str = "reports",
    save_stages: bool = True,
) -> dict:
    """
    Saves each individual diagnostic image from windowed analysis directly to output_dir:
      1. Diagnostic Deficit Map:          {stem}_diagnostic.png
      2. Aligned Baseline CTA Scan:      {stem}_aligned.png
      3. 6-Stage Pipeline Overview:      {stem}_pipeline.png
      4. (Optional) Individual stages:   {stem}_stage1_crop.png ... {stem}_stage6_deficit.png
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_name))[0] or "scan"
    saved = {}

    # 1. Diagnostic Deficit Map
    if final_image is not None and getattr(final_image, "size", 0) > 0:
        diag_path = os.path.join(output_dir, f"{stem}_diagnostic.png")
        cv2.imwrite(diag_path, final_image)
        saved["diagnostic"] = diag_path

    # 2. Baseline Aligned Scan
    if original_aligned is not None and getattr(original_aligned, "size", 0) > 0:
        align_path = os.path.join(output_dir, f"{stem}_aligned.png")
        cv2.imwrite(align_path, original_aligned)
        saved["aligned"] = align_path

    # 3. Scientific Architecture Pipeline Analysis (6-Stage composite)
    if debug_dash is not None and getattr(debug_dash, "size", 0) > 0:
        pipe_path = os.path.join(output_dir, f"{stem}_pipeline.png")
        cv2.imwrite(pipe_path, debug_dash)
        saved["pipeline"] = pipe_path

    # 4. Individual pipeline stage panels (if available on DashboardImage)
    if save_stages and debug_dash is not None:
        panels = getattr(debug_dash, "panels", None)
        if panels and len(panels) > 0:
            stage_slugs = ["crop", "midline", "vessels", "split", "align", "deficit"]
            stage_paths = []
            for idx, panel in enumerate(panels, 1):
                slug = stage_slugs[idx - 1] if idx - 1 < len(stage_slugs) else f"stage{idx}"
                sp = os.path.join(output_dir, f"{stem}_stage{idx}_{slug}.png")
                cv2.imwrite(sp, panel)
                stage_paths.append(sp)
            saved["stages"] = stage_paths

    if saved:
        print(f"[+] Saved windowed report images to {output_dir}\\:")
        for k, v in saved.items():
            if k == "stages":
                print(f"    • Pipeline Stages: {len(v)} individual panel images ({stem}_stage1_crop.png..{stem}_stage{len(v)}.png)")
            else:
                print(f"    • {k.capitalize():<11}: {v}")

    return saved


def show_windowed_report(
    final_image: np.ndarray,
    original_aligned: np.ndarray | None = None,
    debug_dash: np.ndarray | None = None,
    image_name: str = "Scan",
    wait_key: bool = True,
    save_images: bool = True,
    output_dir: str = "reports",
    save_stages: bool = True,
):
    """
    Displays analysis images in separate, clearly titled OpenCV windows
    positioned neatly across the display canvas, and saves each image to the report directory.
    """
    if save_images:
        save_windowed_images(
            final_image=final_image,
            original_aligned=original_aligned,
            debug_dash=debug_dash,
            image_name=image_name,
            output_dir=output_dir,
            save_stages=save_stages,
        )

    windows_created = []

    # Window 1: Diagnostic Overlay (Main Result)
    win1 = f"1. Diagnostic Deficit Map - {image_name}"
    cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win1, 640, 640)
    cv2.moveWindow(win1, 30, 40)
    cv2.imshow(win1, final_image)
    windows_created.append(win1)

    # Window 2: Baseline Aligned Scan
    if original_aligned is not None and original_aligned.size > 0:
        win2 = f"2. Aligned Baseline CTA - {image_name}"
        cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win2, 640, 640)
        cv2.moveWindow(win2, 690, 40)
        cv2.imshow(win2, original_aligned)
        windows_created.append(win2)

    # Window 3: Scientific Architecture Pipeline Analysis (6 Stages: Crop -> Midline -> Vessels -> Split -> Align -> Overlay)
    if debug_dash is not None and debug_dash.size > 0:
        win3 = f"3. Scientific Architecture Pipeline - {image_name}"
        cv2.namedWindow(win3, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win3, 1680, 320)
        cv2.moveWindow(win3, 30, 710)
        cv2.imshow(win3, debug_dash)
        windows_created.append(win3)

    if not wait_key:
        return windows_created

    print(f"\n[Separate Windows Active] Showing {len(windows_created)} individual window(s).")
    print("Press ANY KEY in any image window to advance (or 'q' to abort)...")

    key = None
    while True:
        k = cv2.waitKey(40) & 0xFF
        if k != 255:
            key = k
            break

        # Check if user closed any window
        all_closed = True
        for w in windows_created:
            try:
                if cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) >= 1:
                    all_closed = False
                    break
            except Exception:
                pass
        if all_closed:
            key = ord("q")
            break

    for w in windows_created:
        try:
            cv2.destroyWindow(w)
        except Exception:
            pass
    cv2.waitKey(1)

    return key != ord("q")


# ============================================================
# 3. STANDALONE CLI ENTRYPOINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Window and CLI Report Viewer for Stroke Analysis")
    parser.add_argument("--image", type=str, default=None, help="Scan image path to analyze and view in windowed mode")
    parser.add_argument("--report", type=str, default=None, help="Existing JSON report path to display")
    parser.add_argument("--skip-survey", action="store_true", help="Skip clinical survey")
    parser.add_argument("--no-gui", action="store_true", help="Do not display GUI windows, only print text report")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory to save images (default: reports)")
    parser.add_argument("--no-save-stages", action="store_true", help="Do not save individual pipeline stage panels")

    args = parser.parse_args()

    if args.report:
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
            triage=triage
        )

        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                if args.no_gui:
                    save_windowed_images(
                        final_image=img,
                        image_name=os.path.basename(img_path),
                        output_dir=args.output_dir,
                        save_stages=not args.no_save_stages,
                    )
                else:
                    show_windowed_report(
                        final_image=img,
                        image_name=os.path.basename(img_path),
                        output_dir=args.output_dir,
                        save_stages=not args.no_save_stages,
                        save_images=True,
                    )

    elif args.image:
        from interface import run_inference
        # Run with windowed mode enabled
        run_inference(args.image, skip_survey=args.skip_survey, show_gui=not args.no_gui, report_mode="windows")
    else:
        print("Usage: python report_windowed.py --image data/n1.jpeg [or --report reports/n1_report.json]")


if __name__ == "__main__":
    main()
