import argparse
import json
import os

import cv2

from analyze import calculate_asymmetry, print_asymmetry_profile
from fov_ai import classify_fov
from preprocess import process_image
from report import generate_dashboard_image
from report_windowed import print_cli_report, save_windowed_images, show_windowed_report
from snn_ai import Config, predict_snn
from survey import ClinicalSurvey

# --- Configuration ---
MODEL_WEIGHTS = Config.MODEL_SAVE_PATH

def run_inference(
    image_path,
    mode="auto",
    interactive_fallback=True,
    preview=False,
    show_gui=True,
    min_asym_thresh=50.0,
    bbox_thresh=85.0,
    skip_survey=True,
    report_mode="dashboard",
):
    print("\n==========================================")
    print(f"--- Processing {image_path} ---")
    print("==========================================")

    # 0. FOV DETECTION GATE
    try:
        fov_type, conf = classify_fov(image_path, model_weights="fov_model.pt")
        print(f"Scan FOV: {fov_type} ({conf*100:.1f}% confidence)")
    except Exception as e:
        print(f"[!] FOV classification skipped ({e}). Proceeding with standard detection.")
        fov_type = None

    # 1. PREPROCESS MODULE (Auto with Interactive Fallback)
    result = process_image(
        image_path,
        target_size=Config.INPUT_SIZE,
        mode=mode,
        interactive_fallback=interactive_fallback,
        fov_type=fov_type,
        preview=preview,
    )
    if result is None:
        print(f"[-] Skipped or failed preprocessing for {image_path}")
        return None
    elif result == "QUIT":
        print("[!] User requested quit during alignment.")
        return False

    left_tensor, right_tensor, left_high_res, right_high_res, original_aligned = result
    meta = getattr(result, "meta", {})
    chest_y = meta.get("chest_y")
    tilt_ang = meta.get("angle", 0.0)

    if chest_y is not None and chest_y < original_aligned.shape[0]:
        print(f"[Guillotine] Aortic arch / chest severed at row y={chest_y} to eliminate false positive asymmetry.")
    print(f"[Alignment] Tilt compensation: {tilt_ang:+.2f} deg, Center: ({meta.get('cx')}, {meta.get('cy')})")

    # 2. SNN MODULE (Siamese Neural Network)
    prob = predict_snn(left_tensor, right_tensor, weights_path=MODEL_WEIGHTS)
    if prob is not None:
        snn_class = "STROKE (ASYMMETRIC)" if prob >= 0.50 else "HEALTHY (SYMMETRIC)"
        print(f"AI SNN Classification: {snn_class} [Asymmetry Score: {prob*100:.1f}%, Threshold: 50.0%]")
        print("  (Note: Asymmetry Score is per-scan probability; SNN Cohort Benchmark Accuracy is 93.8%)")
    else:
        print(f"[!] Warning: {MODEL_WEIGHTS} not found. Running morphological analysis only.")

    # 3. ANALYSIS MODULE (Scientific Architecture Pipeline & Deficit Mapping)
    print("Running morphological spatial and mirror reference gap analysis...")
    final_image, debug_dash = calculate_asymmetry(
        original_aligned,
        min_asym_thresh=min_asym_thresh,
        bbox_thresh=bbox_thresh,
        meta=meta,
    )

    # Print clinical asymmetry profile
    if hasattr(debug_dash, "profile") and debug_dash.profile:
        print_asymmetry_profile(debug_dash.profile)

    # Deficit side determination from quantitative anatomical arc profile
    img_affected_side = "NONE"
    snn_tag = "STROKE" if (prob is not None and prob >= 0.50) else "HEALTHY"
    profile = getattr(debug_dash, "profile", None)
    if snn_tag == "STROKE" and profile:
        tot_l = sum(row.get("left_px", 0) for row in profile)
        tot_r = sum(row.get("right_px", 0) for row in profile)
        tot_all = tot_l + tot_r
        ai = (tot_r - tot_l) / float(max(1, tot_all))
        if ai < -0.04:
            img_affected_side = "RIGHT"
        elif ai > 0.04:
            img_affected_side = "LEFT"
        else:
            img_affected_side = "BILATERAL"

    # 4. FINDINGS SUMMARY (Clinical survey hidden for now)
    findings = {
        "affected_side": img_affected_side,
        "snn_prob": prob,
        "snn_class": snn_tag,
        "bbox_count": 0
    }

    triage = {"skipped": True, "summary": "Clinical survey hidden (Pure Radiological Assessment)"}

    # 5. RADIOLOGICAL & AI ASYMMETRY ASSESSMENT (Vision-Only Mode)
    has_imaging_deficit = (prob is not None and prob >= 0.50)

    if has_imaging_deficit:
        if img_affected_side in ("LEFT", "RIGHT"):
            verdict = f"ASYMMETRY DETECTED: {img_affected_side} HEMISPHERE"
        elif img_affected_side == "BILATERAL":
            verdict = "BILATERAL ASYMMETRY DETECTED"
        else:
            verdict = "SIGNIFICANT ASYMMETRY DETECTED"
        banner_color = (30, 30, 220)  # Red in BGR
    else:
        verdict = "SYMMETRICAL SCAN: NEGATIVE FOR SIGNIFICANT DEFICIT"
        banner_color = (40, 180, 50)  # Green in BGR

    snn_str = f"{prob*100:.1f}%" if prob is not None else "N/A"
    sub_banner = f"Deficit Side: {img_affected_side} | SNN Prediction: {snn_tag} ({snn_str} asym)"

    print("\n" + "=" * 65)
    print(f"[RADIOLOGICAL AI VERDICT] {verdict}")
    print(f"  - Status Details: {sub_banner}")
    print(f"  - Radiological Findings: Deficit={img_affected_side}, SNN={snn_tag} ({snn_str})")
    print("=" * 65 + "\n")

    # Draw clinical verdict banner onto Final Diagnostic Output (Light Mode)
    cv2.rectangle(final_image, (0, 0), (final_image.shape[1], 52), (255, 255, 255), -1)
    cv2.line(final_image, (0, 52), (final_image.shape[1], 52), (220, 225, 230), 1)
    cv2.putText(final_image, verdict, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, banner_color, 1, cv2.LINE_AA)
    cv2.putText(final_image, sub_banner, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 65, 75), 1, cv2.LINE_AA)

    # Save diagnostic run report to reports/
    os.makedirs("reports", exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    report_path = os.path.join("reports", f"{stem}_report.json")
    report_data = {
        "image_path": image_path,
        "verdict": verdict,
        "sub_banner": sub_banner,
        "radiological": {
            "deficit_side": img_affected_side,
            "snn_probability": prob,
            "snn_predicted_class": snn_tag,
            "bbox_count": findings["bbox_count"],
            "bounding_boxes": [
                {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"], "asym": b["asymmetry"], "area": b["area"]}
                for b in getattr(debug_dash, "bounding_boxes", [])
            ]
        },
        "clinical_survey": triage
    }
    try:
        with open(report_path, "w") as rf:
            json.dump(report_data, rf, indent=2)
        print(f"[+] Saved diagnostic report to {report_path}")
    except Exception as e:
        print(f"[!] Note: Could not save report JSON ({e})")

    if report_mode == "windows":
        # Print comprehensive clinical report directly to command line
        print_cli_report(
            image_path=image_path,
            verdict=verdict,
            sub_banner=sub_banner,
            findings=findings,
            triage=triage,
            meta=meta,
            debug_dash=debug_dash,
        )
        # Always save windowed diagnostic images directly to reports/
        save_windowed_images(
            final_image=final_image,
            original_aligned=original_aligned,
            debug_dash=debug_dash,
            image_name=image_path,
            output_dir="reports",
            save_stages=True,
        )
        if show_gui:
            proceed = show_windowed_report(
                final_image=final_image,
                original_aligned=original_aligned,
                debug_dash=debug_dash,
                image_name=os.path.basename(image_path),
                wait_key=True,
                save_images=False,
            )
            if not proceed:
                return False
        return True

    # 6. FINAL DASHBOARD IMAGE GENERATION (report.py)
    meta["original_aligned"] = original_aligned
    dashboard_img, _dash_path = generate_dashboard_image(
        image_path=image_path,
        final_image=final_image,
        debug_dash=debug_dash,
        triage=triage,
        verdict=verdict,
        sub_banner=sub_banner,
        findings=findings,
        meta=meta,
        original_aligned=original_aligned,
    )

    # 7. VISUALIZATION OUTPUT
    if show_gui:
        win_name = "Stroke Triage Final Dashboard"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1800, 1050)
        cv2.imshow(win_name, dashboard_img)

        print("\n[Dashboard Ready] Press ANY KEY in the dashboard window to advance (or 'q' to abort)...")
        key = None
        while True:
            k = cv2.waitKey(40) & 0xFF
            if k != 255:
                key = k
                break
            try:
                # Detect if the user clicked the window's 'X' close button
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    key = ord("q")
                    break
            except Exception:
                break

        cv2.destroyAllWindows()
        # Allow Windows DWM time to flush and release window resources cleanly
        cv2.waitKey(1)
        if key == ord("q"):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stroke CTA Analysis Pipeline with Automated Guillotine, Midline Estimator & Clinical Survey"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="Alignment mode: 'auto' runs automated estimator, 'manual' opens interactive joint GUI (default: auto)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable interactive fallback if auto-detection produces low confidence",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="In auto mode, briefly render the auto-detected joints and chest cut in GUI before proceeding",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing CTA scan images (default: 'data')",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Process a single image file instead of entire data directory",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run completely headless without displaying GUI windows (for batch/testing)",
    )
    parser.add_argument(
        "--min-asym",
        type=float,
        default=50.0,
        help="Minimum asymmetry percentage threshold to render deficit overlay (default: 50.0)",
    )
    parser.add_argument(
        "--bbox-thresh",
        type=float,
        default=85.0,
        help="Asymmetry threshold percentage to apply bounding box when really high (default: 85.0)",
    )
    parser.add_argument(
        "--skip-survey",
        action="store_true",
        default=True,
        help="Skip clinical survey (default: True, clinical survey hidden for now)",
    )
    parser.add_argument(
        "--prompt-survey",
        action="store_true",
        help="Prompt interactive clinical triage survey",
    )
    parser.add_argument(
        "--report-mode",
        choices=["dashboard", "windows"],
        default="dashboard",
        help="Display mode: 'dashboard' (all-in-one) or 'windows' (separate windows + CLI text)",
    )
    parser.add_argument(
        "--windows",
        action="store_true",
        help="Shortcut for --report-mode windows",
    )

    args = parser.parse_args()
    interactive_fallback = not args.no_fallback
    show_gui = not args.no_gui
    skip_survey = not args.prompt_survey
    report_mode = "windows" if args.windows else args.report_mode

    if args.image:
        image_files = [args.image]
    else:
        valid_extensions = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = sorted(
            [
                os.path.join(args.data_dir, f)
                for f in os.listdir(args.data_dir)
                if os.path.isfile(os.path.join(args.data_dir, f))
                and os.path.splitext(f)[1].lower() in valid_extensions
            ]
        )

    if not image_files:
        print(f"No scan images found in '{args.data_dir}'")
        return

    print(f"Ready to process {len(image_files)} scan(s) in [{args.mode.upper()}] mode (fallback={interactive_fallback}, skip_survey={skip_survey}, report_mode={report_mode}):")
    for img_p in image_files:
        print(f"  • {img_p}")

    for img_path in image_files:
        status = run_inference(
            img_path,
            mode=args.mode,
            interactive_fallback=interactive_fallback,
            preview=args.preview,
            show_gui=show_gui,
            min_asym_thresh=args.min_asym,
            bbox_thresh=args.bbox_thresh,
            skip_survey=skip_survey,
            report_mode=report_mode,
        )
        if status is False:
            print("User requested loop termination.")
            break


if __name__ == "__main__":
    main()