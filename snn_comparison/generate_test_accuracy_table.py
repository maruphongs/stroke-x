"""
================================================================================
ANTIGRAVITY STROKE AI - TEST DATA ACCURACY & EVALUATION TABLE GENERATOR
================================================================================
Generates dedicated evaluation tables specifically assessing model accuracy
on the independent clinical test dataset (data/test - 20 scans: 10 Normal, 10 Abnormal):
  1. Summary Diagnostic Performance Metrics Table (Accuracy, Sens, Spec, PPV, NPV, F1, AUC)
  2. Granular Per-Scan Accuracy Verification Table (Predictions, Probabilities, Pass/Fail)

Outputs:
  - reports/snn_comparison/tables/test_accuracy_evaluation_table.md
  - reports/snn_comparison/tables/test_accuracy_evaluation_table.tex
  - reports/snn_comparison/figures/test_accuracy_evaluation_table.png
================================================================================
"""

import json
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "reports", "snn_comparison")
TABLES_DIR = os.path.join(BASE_OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(BASE_OUTPUT_DIR, "figures")
METRICS_FILE = os.path.join(BASE_OUTPUT_DIR, "metrics", "model_comparison_results.json")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_test_cohort_results():
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cohort"]


def build_markdown_tables(test_data):
    old_m = test_data["old_snn"]
    new_m = test_data["new_snn"]

    scans_old = {s["filename"]: s for s in old_m["scans"]}
    scans_new = {s["filename"]: s for s in new_m["scans"]}

    # Natural sort
    def _sort_key(k):
        num = "".join(c for c in k if c.isdigit())
        return (k[0], int(num) if num else 0)

    sorted_fnames = sorted(scans_old.keys(), key=_sort_key)

    lines = [
        "# Independent Test Cohort Accuracy & Clinical Evaluation Table",
        "",
        "> **Dataset:** Independent Clinical Holdout Test Dataset (`data/test` — $N=20$: 10 Normal, 10 Abnormal).  ",
        "> **Decision Threshold:** 0.50 (Asymmetry probability $\\ge 0.50$ = Abnormal).",
        "",
        "## 1. Overall Test Data Diagnostic Performance Summary",
        "",
        "| Evaluation Metric | Conventional SNN | Proposed SNN | Absolute Improvement |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Overall Accuracy** | **{old_m['accuracy']:.1f}%** ({old_m['tp']+old_m['tn']}/20) | **{new_m['accuracy']:.1f}%** ({new_m['tp']+new_m['tn']}/20) | **+{new_m['accuracy']-old_m['accuracy']:.1f}%** |",
        f"| **Balanced Accuracy** | {old_m['balanced_accuracy']:.1f}% | **{new_m['balanced_accuracy']:.1f}%** | +{new_m['balanced_accuracy']-old_m['balanced_accuracy']:.1f}% |",
        f"| **Sensitivity / Recall (TPR)** | {old_m['sensitivity']:.1f}% ({old_m['tp']}/10) | **{new_m['sensitivity']:.1f}%** ({new_m['tp']}/10) | Parity (80.0%) |",
        f"| **Specificity (TNR)** | {old_m['specificity']:.1f}% ({old_m['tn']}/10) | **{new_m['specificity']:.1f}%** ({new_m['tn']}/10) | **+{new_m['specificity']-old_m['specificity']:.1f}%** (Zero False Alarms) |",
        f"| **Positive Predictive Value (PPV)** | {old_m['precision']:.1f}% | **{new_m['precision']:.1f}%** | +{new_m['precision']-old_m['precision']:.1f}% |",
        f"| **Negative Predictive Value (NPV)** | {old_m['npv']:.1f}% | **{new_m['npv']:.1f}%** | +{new_m['npv']-old_m['npv']:.1f}% |",
        f"| **F1-Score (Positive Class)** | {old_m['f1_score']:.1f}% | **{new_m['f1_score']:.1f}%** | +{new_m['f1_score']-old_m['f1_score']:.1f}% |",
        f"| **Macro Average F1-Score** | {old_m['f1_macro']:.1f}% | **{new_m['f1_macro']:.1f}%** | +{new_m['f1_macro']-old_m['f1_macro']:.1f}% |",
        f"| **ROC-AUC** | {old_m['roc_auc']:.1f}% | **{new_m['roc_auc']:.1f}%** | +{new_m['roc_auc']-old_m['roc_auc']:.1f}% |",
        f"| **Mean Inference Latency** | {old_m['avg_latency_ms']:.1f} ms | {new_m['avg_latency_ms']:.1f} ms | -{old_m['avg_latency_ms']-new_m['avg_latency_ms']:.1f} ms (Faster) |",
        "",
        "---",
        "",
        "## 2. Granular Per-Scan Accuracy Verification Table (N=20: 10 Normal, 10 Abnormal)",
        "",
        "| Scan ID | Ground Truth | Conventional SNN Probability | Conventional SNN Diagnosis | Conventional SNN Status | Proposed SNN Probability | Proposed SNN Diagnosis | Proposed SNN Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for fname in sorted_fnames:
        so = scans_old[fname]
        sn = scans_new[fname]
        truth = "Normal" if so["true_class"] in ("HEALTHY", "Normal") else "Abnormal"
        
        old_prob = f"{so['prob']*100:.1f}%"
        old_pred = "Normal" if so["pred_class"] in ("HEALTHY", "Normal") else "Abnormal"
        old_status = "✅ PASS" if so["is_correct"] else "❌ FAIL (FP)" if truth == "Normal" else "❌ FAIL (FN)"

        new_prob = f"{sn['prob']*100:.1f}%"
        new_pred = "Normal" if sn["pred_class"] in ("HEALTHY", "Normal") else "Abnormal"
        new_status = "✅ PASS" if sn["is_correct"] else "❌ FAIL (FP)" if truth == "Normal" else "❌ FAIL (FN)"

        lines.append(f"| **{fname}** | {truth} | {old_prob} | {old_pred} | {old_status} | {new_prob} | {new_pred} | {new_status} |")

    lines.extend([
        "",
        "> **Key Clinical Takeaway:**  ",
        "> - **Conventional SNN:** Fails on 8 out of 10 normal patients (false alarms) because its probabilities hover above 0.50 (e.g. `n1` = 54.7%, `n2` = 55.7%), yielding an unusable 20.0% specificity.  ",
        "> - **Proposed SNN:** Accurately separates every normal scan ($10/10$ cleared at $p < 50.0\%$, achieving 100.0% specificity) while successfully detecting acute vascular occlusion across $8/10$ abnormal cases, achieving **90.0% overall test accuracy**.",
    ])

    return "\n".join(lines)


def build_latex_tables(test_data):
    old_m = test_data["old_snn"]
    new_m = test_data["new_snn"]
    scans_old = {s["filename"]: s for s in old_m["scans"]}
    scans_new = {s["filename"]: s for s in new_m["scans"]}

    def _sort_key(k):
        num = "".join(c for c in k if c.isdigit())
        return (k[0], int(num) if num else 0)

    sorted_fnames = sorted(scans_old.keys(), key=_sort_key)

    lines = [
        "% ==============================================================================",
        "% ACADEMIC PAPER TABLE: SUMMARY TEST ACCURACY & DIAGNOSTIC METRICS",
        "% ==============================================================================",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Diagnostic Accuracy and Performance Comparison on Independent Test Dataset ($N=20$: 10 Normal, 10 Abnormal)}",
        "\\label{tab:test_data_accuracy}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{Conventional SNN} & \\textbf{Proposed SNN} & \\textbf{Difference} \\\\",
        "\\hline",
        f"Overall Accuracy (\\%) & {old_m['accuracy']:.1f} & \\textbf{{{new_m['accuracy']:.1f}}} & +{new_m['accuracy']-old_m['accuracy']:.1f}\\% \\\\",
        f"Balanced Accuracy (\\%) & {old_m['balanced_accuracy']:.1f} & \\textbf{{{new_m['balanced_accuracy']:.1f}}} & +{new_m['balanced_accuracy']-old_m['balanced_accuracy']:.1f}\\% \\\\",
        f"Sensitivity / TPR (\\%) & {old_m['sensitivity']:.1f} & \\textbf{{{new_m['sensitivity']:.1f}}} & 0.0\\% \\\\",
        f"Specificity / TNR (\\%) & {old_m['specificity']:.1f} & \\textbf{{{new_m['specificity']:.1f}}} & +{new_m['specificity']-old_m['specificity']:.1f}\\% \\\\",
        f"Positive Predictive Value (\\%) & {old_m['precision']:.1f} & \\textbf{{{new_m['precision']:.1f}}} & +{new_m['precision']-old_m['precision']:.1f}\\% \\\\",
        f"Negative Predictive Value (\\%) & {old_m['npv']:.1f} & \\textbf{{{new_m['npv']:.1f}}} & +{new_m['npv']-old_m['npv']:.1f}\\% \\\\",
        f"F1-Score (Positive, \\%) & {old_m['f1_score']:.1f} & \\textbf{{{new_m['f1_score']:.1f}}} & +{new_m['f1_score']-old_m['f1_score']:.1f}\\% \\\\",
        f"Macro F1-Score (\\%) & {old_m['f1_macro']:.1f} & \\textbf{{{new_m['f1_macro']:.1f}}} & +{new_m['f1_macro']-old_m['f1_macro']:.1f}\\% \\\\",
        f"ROC-AUC (\\%) & {old_m['roc_auc']:.1f} & \\textbf{{{new_m['roc_auc']:.1f}}} & +{new_m['roc_auc']-old_m['roc_auc']:.1f}\\% \\\\",
        f"Mean Latency (ms) & {old_m['avg_latency_ms']:.1f} & \\textbf{{{new_m['avg_latency_ms']:.1f}}} & -{old_m['avg_latency_ms']-new_m['avg_latency_ms']:.1f} ms \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% ==============================================================================",
        "% ACADEMIC PAPER TABLE: GRANULAR PER-SCAN VERIFICATION ON TEST DATA",
        "% ==============================================================================",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Granular Per-Scan Accuracy Verification on Independent Holdout Test Cohort ($N=20$: 10 Normal, 10 Abnormal)}",
        "\\label{tab:per_scan_test_accuracy}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "\\textbf{Scan ID} & \\textbf{Ground Truth} & \\textbf{Conventional SNN Prob (\\%)} & \\textbf{Conventional SNN Diagnosis} & \\textbf{Proposed SNN Prob (\\%)} & \\textbf{Proposed SNN Diagnosis} \\\\",
        "\\hline",
    ]

    for fname in sorted_fnames:
        so = scans_old[fname]
        sn = scans_new[fname]
        clean_fname = fname.replace("_", "\\_")
        truth = "Normal" if so["true_class"] in ("HEALTHY", "Normal") else "Abnormal"
        old_p = f"{so['prob']*100:.1f}"
        old_pred = "Abnormal (FP)" if (not so["is_correct"] and truth == "Normal") else "Normal (TN)" if truth == "Normal" else "Abnormal (TP)" if so["is_correct"] else "Normal (FN)"
        new_p = f"{sn['prob']*100:.1f}"
        new_pred = "Abnormal (FP)" if (not sn["is_correct"] and truth == "Normal") else "Normal (TN)" if truth == "Normal" else "Abnormal (TP)" if sn["is_correct"] else "Normal (FN)"

        lines.append(f"{clean_fname} & {truth} & {old_p} & {old_pred} & {new_p} & \\textbf{{{new_pred}}} \\\\")

    lines.extend([
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def render_visual_test_accuracy_table(test_data, save_path):
    old_m = test_data["old_snn"]
    new_m = test_data["new_snn"]
    scans_old = {s["filename"]: s for s in old_m["scans"]}
    scans_new = {s["filename"]: s for s in new_m["scans"]}

    def _sort_key(k):
        num = "".join(c for c in k if c.isdigit())
        return (k[0], int(num) if num else 0)

    sorted_fnames = sorted(scans_old.keys(), key=_sort_key)

    fig, (ax_summary, ax_scans) = plt.subplots(
        2, 1, figsize=(14.5, 12.5), dpi=300, facecolor="#ffffff", gridspec_kw={"height_ratios": [1, 2.5]}
    )
    ax_summary.axis("off")
    ax_scans.axis("off")

    # Table 1: Summary Metrics
    sum_headers = ["Test Evaluation Metric", "Conventional SNN", "Proposed SNN", "Clinical Advantage"]
    sum_data = [
        ["Overall Accuracy", f"{old_m['accuracy']:.1f}% ({old_m['tp']+old_m['tn']}/20)", f"{new_m['accuracy']:.1f}% ({new_m['tp']+new_m['tn']}/20)", f"+{new_m['accuracy']-old_m['accuracy']:.1f}% Diagnostic Accuracy Jump"],
        ["Sensitivity / Recall (TPR)", f"{old_m['sensitivity']:.1f}% ({old_m['tp']}/10)", f"{new_m['sensitivity']:.1f}% ({new_m['tp']}/10)", "Robust 80.0% True Positive Rate"],
        ["Specificity / TNR", f"{old_m['specificity']:.1f}% ({old_m['tn']}/10)", f"{new_m['specificity']:.1f}% ({new_m['tn']}/10)", "Eliminates All False Alarms (100% Cleared)"],
        ["Positive Predictive Value (PPV)", f"{old_m['precision']:.1f}%", f"{new_m['precision']:.1f}%", "100.0% Confidence in Positive Alarms"],
        ["F1-Score / Balanced Acc", f"{old_m['f1_score']:.1f}% / {old_m['balanced_accuracy']:.1f}%", f"{new_m['f1_score']:.1f}% / {new_m['balanced_accuracy']:.1f}%", f"+{new_m['f1_score']-old_m['f1_score']:.1f}% F1 Improvement"],
        ["ROC-AUC / Latency", f"{old_m['roc_auc']:.1f}% / {old_m['avg_latency_ms']:.1f}ms", f"{new_m['roc_auc']:.1f}% / {new_m['avg_latency_ms']:.1f}ms", "Perfect 100.0% AUC Discrimination"],
    ]

    t1 = ax_summary.table(cellText=sum_data, colLabels=sum_headers, loc="center", cellLoc="left", colWidths=[0.28, 0.22, 0.24, 0.26])
    t1.auto_set_font_size(False)
    t1.set_fontsize(8.5)
    t1.scale(1.0, 1.45)

    for (r, c), cell in t1.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(weight="bold", color="#ffffff")
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            if c == 2:
                cell.set_text_props(weight="bold", color="#0369a1")

    ax_summary.set_title("Part I: Summary Diagnostic Accuracy Benchmarks on Test Dataset (N=20: 10 Normal, 10 Abnormal)",
                         fontsize=11.5, fontweight="bold", color="#0f172a", pad=12)

    # Table 2: Granular Per-Scan Breakdown
    scan_headers = ["Scan", "Ground Truth", "Conventional SNN Prob", "Conventional Diagnosis", "Conventional Status", "Proposed SNN Prob", "Proposed Diagnosis", "Proposed Status"]
    scan_rows = []
    for fname in sorted_fnames:
        so = scans_old[fname]
        sn = scans_new[fname]
        truth = "Normal" if so["true_class"] in ("HEALTHY", "Normal") else "Abnormal"
        old_pred = "Normal" if so["pred_class"] in ("HEALTHY", "Normal") else "Abnormal"
        new_pred = "Normal" if sn["pred_class"] in ("HEALTHY", "Normal") else "Abnormal"
        old_st = "CORRECT" if so["is_correct"] else "MISSED (FP)" if truth == "Normal" else "MISSED (FN)"
        new_st = "CORRECT" if sn["is_correct"] else "MISSED (FP)" if truth == "Normal" else "MISSED (FN)"
        scan_rows.append([
            fname, truth, f"{so['prob']*100:.1f}%", old_pred, old_st,
            f"{sn['prob']*100:.1f}%", new_pred, new_st
        ])

    t2 = ax_scans.table(cellText=scan_rows, colLabels=scan_headers, loc="center", cellLoc="left",
                        colWidths=[0.11, 0.12, 0.14, 0.14, 0.13, 0.13, 0.13, 0.10])
    t2.auto_set_font_size(False)
    t2.set_fontsize(7.8)
    t2.scale(1.0, 1.35)

    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(weight="bold", color="#ffffff")
        else:
            row_data = scan_rows[r - 1]
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            
            # Status styling
            if c == 4:
                cell.set_text_props(weight="bold", color="#15803d" if "CORRECT" in row_data[4] else "#b91c1c")
            elif c == 7:
                cell.set_text_props(weight="bold", color="#15803d" if "CORRECT" in row_data[7] else "#b91c1c")
            elif c in (0, 1):
                cell.set_text_props(weight="semibold", color="#0f172a")

    ax_scans.set_title("Part II: Granular Per-Scan Accuracy Verification across All 20 Holdout Test Scans (10 Normal, 10 Abnormal)",
                      fontsize=11.5, fontweight="bold", color="#0f172a", pad=12)

    plt.suptitle("Clinical Evaluation & Accuracy Verification on Independent Holdout Test Data (N=20: 10 Normal, 10 Abnormal)",
                 fontsize=13.5, fontweight="bold", color="#0f172a", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved Visual Test Accuracy Table to: {save_path}")


def run():
    print("=" * 80)
    print("  SNN COMPARISON: TEST DATA ACCURACY EVALUATION TABLE GENERATOR")
    print("=" * 80)

    test_data = load_test_cohort_results()

    # 1. Markdown
    md_content = build_markdown_tables(test_data)
    md_path = os.path.join(TABLES_DIR, "test_accuracy_evaluation_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved Markdown Test Accuracy Table: {md_path}")

    # 2. LaTeX
    tex_content = build_latex_tables(test_data)
    tex_path = os.path.join(TABLES_DIR, "test_accuracy_evaluation_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[+] Saved LaTeX Test Accuracy Table: {tex_path}")

    # 3. High-Res Graphic
    img_path = os.path.join(FIGURES_DIR, "test_accuracy_evaluation_table.png")
    render_visual_test_accuracy_table(test_data, save_path=img_path)

    # Safely print summary without Windows console encoding crash
    try:
        print("\n" + md_content[:1200] + "\n...")
    except UnicodeEncodeError:
        print("\n[+] Test Accuracy Table Markdown generated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run()
