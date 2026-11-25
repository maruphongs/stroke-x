import argparse
import json
import os
import sys
import time

# Import core pipeline modules
from interface import run_inference


def discover_scans(data_dir: str, pattern: str | None = None) -> list[str]:
    """
    Discovers and sorts all valid CTA scan images in data_dir and its subdirectories.
    Prioritizes data/test or direct scans while ignoring cache and non-test datasets.
    """
    valid_exts = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
    candidates = []

    # If data_dir has a dedicated 'test' folder and no direct scans, prefer test folder
    test_sub = os.path.join(data_dir, "test")
    search_root = test_sub if os.path.isdir(test_sub) else data_dir

    skip_dirs = {"cache_snn", "reports", ".old", ".venv", "scratch", "__pycache__"}
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        root_lower = root.lower().replace("\\", "/")
        if "fov_dataset" in root_lower:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                if pattern and (pattern.lower() not in f.lower()):
                    continue
                candidates.append(os.path.join(root, f))

    # Natural sort: n1..n10, s1..s10
    def natural_sort_key(p):
        fname = os.path.basename(p)
        stem = os.path.splitext(fname)[0]
        prefix = stem[0].lower() if len(stem) > 0 else ""
        num_part = "".join([c for c in stem[1:] if c.isdigit()])
        num = int(num_part) if num_part else 0
        return (prefix, num, fname)

    return sorted(set(candidates), key=natural_sort_key)


def get_expected_label(filename: str) -> str:
    """
    Identifies expected clinical cohort by prefix:
    'n*' -> NORMAL
    's*' -> STROKE
    """
    base = os.path.basename(filename).lower()
    if base.startswith("n"):
        return "NORMAL"
    elif base.startswith("s"):
        return "STROKE"
    return "UNKNOWN"


def run_batch_tests(
    data_dir: str = "data",
    pattern: str | None = None,
    mode: str = "auto",
    report_mode: str = "windows",
    show_gui: bool = True,
    verbose: bool = False,
    fail_fast: bool = False,
    save_json: str | None = None,
) -> int:
    """
    Executes automated end-to-end tests on all images in data_dir.
    Returns 0 if all tests succeed, 1 if any failure occurs.
    """
    if not os.path.isdir(data_dir):
        print(f"[!] Error: Data directory '{data_dir}' does not exist.")
        return 1

    scan_files = discover_scans(data_dir, pattern=pattern)
    if not scan_files:
        print(f"[!] No scan images found in '{data_dir}' matching filter '{pattern or '*'}'")
        return 1

    print("=" * 85)
    print("       ANTIGRAVITY STROKE AI TEST SUITE - AUTOMATED DATA COHORT RUNNER")
    print("=" * 85)
    print(f" Target Directory : {os.path.abspath(data_dir)}")
    print(f" Execution Mode   : {mode.upper()} ({'GUI Windows Display' if show_gui else 'Headless / No-GUI'})")
    print(f" Report Mode      : {report_mode.upper()}")
    print("-" * 85)

    results = []
    start_total_time = time.time()

    # Track classification metrics
    tp, fp, tn, fn = 0, 0, 0, 0
    snn_tp, snn_fp, snn_tn, snn_fn = 0, 0, 0, 0
    pipeline_passes = 0
    pipeline_fails = 0

    for idx, scan_path in enumerate(scan_files, 1):
        stem = os.path.splitext(os.path.basename(scan_path))[0]
        expected = get_expected_label(scan_path)
        print(f"[{idx:2d}/{len(scan_files)}] Testing {os.path.basename(scan_path):<12} (Expected: {expected:<6}) ... ", end="", flush=True)

        t0 = time.time()
        error_msg = None
        report_data = None
        status = None

        # If show_gui or verbose is True, don't suppress stdout so user sees CLI reports in real-time
        suppress_stdout = not verbose and not show_gui
        if suppress_stdout:
            saved_stdout = sys.stdout
            from io import StringIO
            captured_out = StringIO()
            sys.stdout = captured_out

        try:
            status = run_inference(
                scan_path,
                mode=mode,
                interactive_fallback=show_gui,
                preview=False,
                show_gui=show_gui,
                min_asym_thresh=50.0,
                bbox_thresh=85.0,
                skip_survey=True,
                report_mode=report_mode,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}"
            status = False
        finally:
            if suppress_stdout:
                sys.stdout = saved_stdout

        elapsed = time.time() - t0

        # Load generated report JSON to inspect metrics
        report_json_path = os.path.join("reports", f"{stem}_report.json")
        if os.path.exists(report_json_path):
            try:
                with open(report_json_path, "r") as rf:
                    report_data = json.load(rf)
            except Exception:
                pass

        # Determine diagnostic metrics
        if report_data:
            verdict = report_data.get("verdict", "UNKNOWN")
            rad = report_data.get("radiological", {})
            deficit_side = rad.get("deficit_side", "NONE")
            snn_prob = rad.get("snn_probability")
            bbox_count = rad.get("bbox_count", 0)
        else:
            verdict = "PIPELINE_ERROR"
            deficit_side = "ERROR"
            snn_prob = None
            bbox_count = 0

        # 1. SNN Model Standalone Classification concordance
        snn_stroke_pred = (snn_prob is not None and snn_prob >= 0.50)
        snn_class = "STROKE" if snn_stroke_pred else "NORMAL"
        if expected == "NORMAL":
            if not snn_stroke_pred:
                snn_tn += 1
                snn_correct = True
            else:
                snn_fp += 1
                snn_correct = False
        elif expected == "STROKE":
            if snn_stroke_pred:
                snn_tp += 1
                snn_correct = True
            else:
                snn_fn += 1
                snn_correct = False
        else:
            snn_correct = None

        # 2. Full Diagnostic Pipeline concordance (Vision Morphological Deficits + SNN Asymmetry Fusion)
        is_stroke_predicted = (
            ("ASYMMETRY DETECTED" in verdict)
            or (deficit_side in ("LEFT", "RIGHT", "BILATERAL"))
            or snn_stroke_pred
        )

        if expected == "NORMAL":
            if not is_stroke_predicted:
                tn += 1
                diag_correct = True
            else:
                fp += 1
                diag_correct = False
        elif expected == "STROKE":
            if is_stroke_predicted:
                tp += 1
                diag_correct = True
            else:
                fn += 1
                diag_correct = False
        else:
            diag_correct = None

        if status is not False and status is not None and error_msg is None:
            pipeline_passes += 1
            test_pass = True
            tag = "PASS"
        else:
            pipeline_fails += 1
            test_pass = False
            tag = "FAIL"

        snn_str = f"{snn_prob*100:.1f}%" if snn_prob is not None else "N/A"
        diag_icon = "[OK]" if diag_correct is True else ("[MISS]" if diag_correct is False else "[?]")
        snn_icon = "[SNN:OK]" if snn_correct is True else ("[SNN:MISS]" if snn_correct is False else "[SNN:?]")

        print(f"{tag} ({elapsed:4.2f}s) | Verdict: {verdict.split(':')[0]:<7} | Deficit: {deficit_side:<5} | BBoxes: {bbox_count} | SNN: {snn_str} {snn_icon} | Pipeline {diag_icon}")

        if error_msg and verbose:
            print(f"    --> Error: {error_msg}")

        results.append({
            "scan": os.path.basename(scan_path),
            "path": scan_path,
            "expected": expected,
            "pipeline_status": tag,
            "elapsed_seconds": round(elapsed, 3),
            "verdict": verdict,
            "deficit_side": deficit_side,
            "snn_probability": snn_prob,
            "snn_predicted_class": snn_class,
            "snn_concordant": snn_correct,
            "bbox_count": bbox_count,
            "diagnostic_concordant": diag_correct,
            "error": error_msg,
        })

        if fail_fast and not test_pass:
            print(f"\n[!] Fail-fast triggered on {scan_path}.")
            break

    total_elapsed = time.time() - start_total_time
    avg_elapsed = total_elapsed / len(results) if results else 0.0

    # Print summary table
    print("\n" + "=" * 98)
    print("                                   TEST SUMMARY MATRIX")
    print("=" * 98)
    header = f"{'Scan':<11} | {'Expected':<8} | {'Status':<6} | {'Time':<6} | {'Deficit':<7} | {'BBoxes':<6} | {'SNN Prob':<8} | {'SNN Class':<9} | {'Verdict'}"
    print(header)
    print("-" * 98)
    for r in results:
        s_prob = f"{r['snn_probability']*100:.1f}%" if r['snn_probability'] is not None else "N/A"
        s_cls = f"{r['snn_predicted_class'][:4]} {'[OK]' if r['snn_concordant'] else '[X]'}"
        v_short = r['verdict'].split('(')[0].strip() if r['verdict'] else "N/A"
        if len(v_short) > 24:
            v_short = v_short[:21] + "..."
        row = (
            f"{r['scan']:<11} | "
            f"{r['expected']:<8} | "
            f"{r['pipeline_status']:<6} | "
            f"{r['elapsed_seconds']:5.2f}s | "
            f"{r['deficit_side']:<7} | "
            f"{r['bbox_count']:<6} | "
            f"{s_prob:<8} | "
            f"{s_cls:<9} | "
            f"{v_short}"
        )
        print(row)
    print("-" * 98)

    # Diagnostic Metrics
    total_scans = len(results)
    total_evaluated = tp + fp + tn + fn
    sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    accuracy = ((tp + tn) / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    # SNN AI Standalone Metrics
    snn_evaluated = snn_tp + snn_fp + snn_tn + snn_fn
    snn_sensitivity = (snn_tp / (snn_tp + snn_fn) * 100.0) if (snn_tp + snn_fn) > 0 else 0.0
    snn_specificity = (snn_tn / (snn_tn + snn_fp) * 100.0) if (snn_tn + snn_fp) > 0 else 0.0
    snn_accuracy = ((snn_tp + snn_tn) / snn_evaluated * 100.0) if snn_evaluated > 0 else 0.0

    print(" PERFORMANCE BENCHMARKS:")
    print(" -----------------------------------------------------------------------------------")
    print(" 1. SNN AI MODEL BENCHMARKS (Siamese ResNet-18 @ Decision Threshold = 0.50):")
    print(f"    - SNN Overall Accuracy   : {snn_accuracy:.1f}% ({snn_tp+snn_tn}/{snn_evaluated})")
    print(f"    - SNN Cohort Sensitivity : {snn_sensitivity:.1f}% (TP={snn_tp}, FN={snn_fn})")
    print(f"    - SNN Cohort Specificity : {snn_specificity:.1f}% (TN={snn_tn}, FP={snn_fp})")
    print(" -----------------------------------------------------------------------------------")
    print(" 2. FULL DIAGNOSTIC PIPELINE BENCHMARKS (Vision Morphological Deficit + SNN Fusion):")
    print(f"    - Diagnostic Accuracy    : {accuracy:.1f}% ({tp+tn}/{total_evaluated})")
    print(f"    - Cohort Sensitivity     : {sensitivity:.1f}% (TP={tp}, FN={fn})")
    print(f"    - Cohort Specificity     : {specificity:.1f}% (TN={tn}, FP={fp})")
    print(f"    - Pipeline Executions    : {pipeline_passes}/{total_scans} Succeeded ({(pipeline_passes/total_scans*100.0):.1f}%)")
    if pipeline_fails > 0:
        print(f"    - Pipeline Failures      : {pipeline_fails}")
    print(f"    - Total Execution Time   : {total_elapsed:.2f} seconds (Avg: {avg_elapsed:.2f}s/scan)")
    print("=" * 98)

    # Save JSON report if requested
    if save_json:
        os.makedirs(os.path.dirname(os.path.abspath(save_json)) if os.path.dirname(save_json) else ".", exist_ok=True)
        summary_payload = {
            "total_scans": total_scans,
            "pipeline_passed": pipeline_passes,
            "pipeline_failed": pipeline_fails,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "avg_seconds_per_scan": round(avg_elapsed, 2),
            "snn_metrics": {
                "decision_threshold": 0.50,
                "tp": snn_tp, "fp": snn_fp, "tn": snn_tn, "fn": snn_fn,
                "sensitivity_pct": round(snn_sensitivity, 2),
                "specificity_pct": round(snn_specificity, 2),
                "accuracy_pct": round(snn_accuracy, 2),
            },
            "diagnostic_pipeline_metrics": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "sensitivity_pct": round(sensitivity, 2),
                "specificity_pct": round(specificity, 2),
                "accuracy_pct": round(accuracy, 2),
            },
            "scans": results
        }
        try:
            with open(save_json, "w") as jf:
                json.dump(summary_payload, jf, indent=2)
            print(f"[+] Detailed test summary saved to {save_json}")
        except Exception as e:
            print(f"[!] Warning: Could not write JSON summary ({e})")

    return 0 if pipeline_fails == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Automated Test Suite for Stroke AI CTA Data Cohort"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing test scan images (default: 'data')",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter scan filenames by substring/prefix (e.g. 'n' for normals, 's' for stroke, or 's4')",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="Alignment mode to test (default: 'auto')",
    )
    parser.add_argument(
        "--report-mode",
        choices=["dashboard", "windows"],
        default="windows",
        help="Report visualization mode to generate during test: 'windows' (default) or 'dashboard'",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Display OpenCV GUI windows during testing (default: headless / no-gui)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless without displaying OpenCV GUI windows (default)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="reports/test_results.json",
        help="Path to save JSON test results (default: 'reports/test_results.json')",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any scan fails pipeline execution",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show complete stdout logs from every pipeline stage",
    )

    args = parser.parse_args()
    show_gui = args.gui and not args.no_gui
    exit_code = run_batch_tests(
        data_dir=args.data_dir,
        pattern=args.filter,
        mode=args.mode,
        report_mode=args.report_mode,
        show_gui=show_gui,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        save_json=args.save_json,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
