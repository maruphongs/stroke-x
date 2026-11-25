import json
import os
import sys


def safe_prompt_input(prompt_text=""):
    """
    Reads terminal input while ensuring any open OpenCV windows remain active
    and responsive without freezing or crashing when switching focus.
    """
    print(prompt_text, end="", flush=True)
    try:
        sys.stdin.fileno()
        is_real_tty = sys.stdin.isatty()
    except Exception:
        is_real_tty = False

    if not is_real_tty:
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\r\n")

    try:
        import msvcrt

        import cv2
        chars = []
        while True:
            # Pump window events every 20ms to prevent Windows (Not Responding) hang
            cv2.waitKey(20)
            if msvcrt.kbhit():
                ch = msvcrt.getwche()
                if ch in ("\r", "\n"):
                    print()
                    return "".join(chars)
                elif ch == "\b":
                    if chars:
                        chars.pop()
                        sys.stdout.write(" \b")
                        sys.stdout.flush()
                elif ch == "\x03":
                    raise KeyboardInterrupt
                elif ord(ch) >= 32:
                    chars.append(ch)
    except Exception:
        return input().rstrip("\r\n")


class ClinicalSurvey:
    @staticmethod
    def evaluate(survey_data: dict) -> dict:
        """
        Evaluates clinical questionnaire and calculates:
          - Survey status (Completed vs Skipped)
          - Stroke mimic disqualification (hypoglycemia <50 mg/dL)
          - Clinical LVO risk tier (via VAN protocol and NIHSS)
          - Contralateral expected cerebral hemisphere lesion (pyramidal decussation)
          - Interventional time window eligibility (IVT <=4.5h, EVT <=24h)
        """
        if not survey_data or survey_data.get("skipped", False):
            return {
                "skipped": True,
                "eligible": True,
                "mimic_alert": None,
                "suspected_hemisphere": "UNKNOWN",
                "lvo_risk": "UNSPECIFIED",
                "window": "UNSPECIFIED",
                "blood_glucose_mg_dl": None,
                "lkw_hours": None,
                "weakness_side": "NONE",
                "nihss_estimate": 0,
                "summary": "Clinical survey skipped (Pure Radiological Assessment)"
            }

        glucose = float(survey_data.get("blood_glucose_mg_dl", 110.0))
        lkw_hours = float(survey_data.get("lkw_hours", 2.0))
        weakness = str(survey_data.get("weakness_side", "NONE")).strip().upper()
        vision = bool(survey_data.get("vision_cut", False))
        aphasia = bool(survey_data.get("aphasia", False))
        neglect = bool(survey_data.get("neglect", False))
        nihss = int(survey_data.get("nihss_estimate", 0))

        report = {
            "skipped": False,
            "eligible": True,
            "mimic_alert": None,
            "suspected_hemisphere": "UNKNOWN",
            "lvo_risk": "LOW",
            "window": "OUT_OF_WINDOW",
            "blood_glucose_mg_dl": glucose,
            "lkw_hours": lkw_hours,
            "weakness_side": weakness,
            "nihss_estimate": nihss,
            "summary": ""
        }

        # 1. Hypoglycemia rule-out
        if glucose < 50.0:
            report["eligible"] = False
            report["mimic_alert"] = (
                f"CRITICAL: Capillary glucose is {glucose:.0f} mg/dL (<50 mg/dL). "
                "Stroke mimic suspected! Correct hypoglycemia immediately."
            )

        # 2. Reperfusion window eligibility
        if lkw_hours <= 4.5:
            report["window"] = "IVT & EVT Window (<=4.5h)"
        elif lkw_hours <= 24.0:
            report["window"] = "EVT Window (4.5h - 24h)"
        else:
            report["window"] = "Late Presentation (>24h)"

        # 3. Contralateral Brain Localization (Pyramidal Decussation)
        if weakness == "LEFT":
            report["suspected_hemisphere"] = "RIGHT"
        elif weakness == "RIGHT":
            report["suspected_hemisphere"] = "LEFT"

        # Cortical localization overrides/supplements if motor weakness is ambiguous
        if report["suspected_hemisphere"] == "UNKNOWN":
            if aphasia:
                report["suspected_hemisphere"] = "LEFT"   # Dominant hemisphere
            elif neglect:
                report["suspected_hemisphere"] = "RIGHT"  # Non-dominant hemisphere

        # 4. VAN LVO Protocol
        has_motor = weakness in ("LEFT", "RIGHT")
        has_cortical = vision or aphasia or neglect

        if (has_motor and has_cortical) or nihss >= 10:
            report["lvo_risk"] = "HIGH"
        elif has_motor or nihss >= 5:
            report["lvo_risk"] = "MODERATE"
        else:
            report["lvo_risk"] = "LOW"

        # Summary text
        report["summary"] = (
            f"LVO Risk: {report['lvo_risk']} | "
            f"Suspected Brain: {report['suspected_hemisphere']} | "
            f"Window: {report['window']}"
        )
        if report["mimic_alert"]:
            report["summary"] += f" | {report['mimic_alert']}"

        return report

    @staticmethod
    def prompt_survey(
        image_path: str,
        findings: dict | None = None,
        auto_skip: bool = False
    ):
        """
        Interactive clinical survey prompted AFTER image analysis.
        Allows the user to skip the survey by pressing Enter or typing 's'/'skip'.
        Returns:
            dict with survey fields (or {"skipped": True})
            or "QUIT" if user requested abort
        """
        # Check for auto-skip flag or non-interactive terminal
        if auto_skip:
            return {"skipped": True}

        # Check for existing survey JSON
        json_path = os.path.splitext(image_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    saved_data = json.load(f)
                print(f"\n[Clinical Survey] Found saved clinical record for {os.path.basename(image_path)}:")
                print(f"  • Glucose: {saved_data.get('blood_glucose_mg_dl', 'N/A')} mg/dL | LKW: {saved_data.get('lkw_hours', 'N/A')}h")
                print(f"  • Weakness: {saved_data.get('weakness_side', 'NONE')} | NIHSS: {saved_data.get('nihss_estimate', 0)}")

                if not sys.stdin.isatty():
                    return saved_data

                ans = safe_prompt_input("  Use saved clinical data? [Y = Use saved / e = Edit / s = Skip]: ").strip().lower()
                if ans in ("s", "skip"):
                    print("[Clinical Survey] Survey skipped by user.")
                    return {"skipped": True}
                elif ans in ("q", "quit"):
                    return "QUIT"
                elif ans not in ("e", "edit"):
                    saved_data["skipped"] = False
                    return saved_data
            except Exception as e:
                print(f"[!] Note: Could not read existing survey JSON ({e}).")

        # Display Survey Header & Image Findings
        print("\n" + "=" * 65)
        print(f" CLINICAL TRIAGE SURVEY -- {os.path.basename(image_path)}")
        print("=" * 65)
        if findings:
            aff = findings.get("affected_side", "NONE")
            prob = findings.get("snn_prob")
            n_boxes = findings.get("bbox_count", 0)
            snn_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"
            print("  [Radiological Findings]")
            print(f"   • Deficit Side         : {aff} hemisphere")
            print(f"   • Flagged Deficits     : {n_boxes} high-deficit region(s)")
            snn_cls = "STROKE" if (prob is not None and prob >= 0.50) else "HEALTHY"
            print(f"   • AI Model Prediction  : {snn_cls} (Asymmetry Prob: {snn_str})")
            print("-" * 65)

        print("  Options:")
        print("    [Y] Complete clinical survey (7 quick questions)")
        print("    [S] Skip survey (Default: press ENTER to proceed without survey)")
        print("    [Q] Quit analysis")
        print("-" * 65)

        try:
            choice = safe_prompt_input("Enter choice [y/S/q] (default: Skip): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[Clinical Survey] Input interrupted. Skipping survey.")
            return {"skipped": True}

        if choice in ("q", "quit"):
            return "QUIT"

        if choice not in ("y", "yes"):
            print("[Clinical Survey] Survey skipped. Proceeding with radiological assessment.")
            return {"skipped": True}

        # User elected to answer questionnaire
        print("\n--- Enter Clinical Triage Parameters (press ENTER for default) ---")

        def _get_float(prompt_text, default_val, min_val=0.0, max_val=1000.0):
            while True:
                try:
                    val = safe_prompt_input(f"  • {prompt_text} [default: {default_val}]: ").strip()
                    if not val:
                        return default_val
                    if val.lower() in ("skip", "s"):
                        return default_val
                    num = float(val)
                    if min_val <= num <= max_val:
                        return num
                    print(f"    [!] Value must be between {min_val} and {max_val}.")
                except ValueError:
                    print("    [!] Please enter a valid number.")

        def _get_bool(prompt_text, default_val=False):
            def_str = "y" if default_val else "n"
            val = safe_prompt_input(f"  • {prompt_text} (y/n) [default: {def_str}]: ").strip().lower()
            if not val:
                return default_val
            return val in ("y", "yes", "true", "1")

        def _get_side(prompt_text, default_val="NONE"):
            while True:
                val = safe_prompt_input(f"  • {prompt_text} (left/right/none) [default: {default_val}]: ").strip().upper()
                if not val:
                    return default_val
                if val in ("L", "LEFT"):
                    return "LEFT"
                elif val in ("R", "RIGHT"):
                    return "RIGHT"
                elif val in ("N", "NONE", "NO"):
                    return "NONE"
                print("    [!] Please enter 'left', 'right', or 'none'.")

        try:
            glucose = _get_float("Capillary Glucose (mg/dL; <50 is mimic)", default_val=110.0, min_val=10.0, max_val=800.0)
            lkw = _get_float("Last Known Well (Hours ago; <=4.5h IVT, <=24h EVT)", default_val=2.0, min_val=0.0, max_val=168.0)
            weakness = _get_side("Motor Weakness Extremity", default_val="NONE")
            vision = _get_bool("Vision Cut / Conjugate Gaze Deviation", default_val=False)
            aphasia = _get_bool("Aphasia / Language Deficit / Dysarthria", default_val=False)
            neglect = _get_bool("Hemispatial Neglect / Extinction", default_val=False)
            nihss = int(_get_float("Estimated NIHSS Score (0 - 42)", default_val=0, min_val=0, max_val=42))

            survey_data = {
                "skipped": False,
                "blood_glucose_mg_dl": glucose,
                "lkw_hours": lkw,
                "weakness_side": weakness,
                "vision_cut": vision,
                "aphasia": aphasia,
                "neglect": neglect,
                "nihss_estimate": nihss
            }

            print(f"\n[Survey Recorded] Glucose: {glucose:.0f} mg/dL | LKW: {lkw:.1f}h | Weakness: {weakness} | NIHSS: {nihss}")

            # Automatically save survey record to JSON without prompting
            try:
                with open(json_path, "w") as f:
                    json.dump(survey_data, f, indent=2)
                print(f"  [+] Clinical survey saved to {json_path}")
            except Exception as e:
                print(f"  [!] Note: Could not write survey JSON ({e})")

            return survey_data

        except (EOFError, KeyboardInterrupt):
            print("\n[Clinical Survey] Survey aborted. Proceeding with defaults.")
            return {"skipped": True}

    @staticmethod
    def load_or_prompt(image_path: str, interactive: bool = False) -> dict:
        """Backwards-compatibility wrapper."""
        if not interactive:
            json_path = os.path.splitext(image_path)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        return json.load(f)
                except Exception:
                    pass
            return {"skipped": True}
        return ClinicalSurvey.prompt_survey(image_path, auto_skip=False)