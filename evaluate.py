#!/usr/bin/env python3
"""
================================================================================
ANTIGRAVITY STROKE AI - MODEL EVALUATION SUITE
================================================================================
Comprehensive evaluation module for:
  1. FOV AI (Field-of-View Classifier: MobileNetV3-Small)
  2. SNN AI (Siamese Neural Network Asymmetry Detector: ResNet-18)

Evaluates models using identical training datasets due to limited clinical scans.
Generates:
  - Formatted Training Configuration Tables
  - Performance Metrics Tables (Accuracy, Balanced Acc, Sensitivity, Specificity, etc.)
  - High-Resolution Confusion Matrix Visualizations (Heatmaps & Text Matrices)
  - Comparative Performance Visualizations and JSON / Markdown Reports
================================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Project imports
from fov_ai import classify_fov, get_fov_model, preprocess_for_fov
from snn_ai import (
    Config as SNNConfig,
    discover_scans,
    load_cohort_with_cache,
    load_snn_model,
    predict_snn,
)

# Output directories
DEFAULT_REPORTS_DIR = "reports"
os.makedirs(DEFAULT_REPORTS_DIR, exist_ok=True)


# Configure stdout / stderr for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==============================================================================
# 1. FORMATTING & TABLE UTILITIES
# ==============================================================================

def format_title_banner(title: str, width: int = 88) -> str:
    """Creates a stylized header banner."""
    sep = "=" * width
    return f"\n{sep}\n  {title.center(width - 4)}\n{sep}"


def format_sub_banner(title: str, width: int = 88) -> str:
    """Creates a stylized sub-header banner."""
    sep = "-" * width
    return f"\n{sep}\n  {title.center(width - 4)}\n{sep}"


def print_box_table(title: str, headers: List[str], rows: List[List[str]], col_widths: List[int] = None):
    """
    Renders a clean ASCII table in the terminal compatible with all platforms.
    """
    if not rows:
        print(f"[!] No data to display for {title}")
        return

    num_cols = len(headers)
    if col_widths is None:
        col_widths = []
        for i in range(num_cols):
            max_len = len(headers[i])
            for r in rows:
                if i < len(r):
                    max_len = max(max_len, len(str(r[i])))
            col_widths.append(max_len + 2)

    top_border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_sep = "+" + "+".join("=" * (w + 2) for w in col_widths) + "+"
    row_sep    = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    bot_border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    print(f"\n{title}:")
    print(top_border)

    # Header row
    hdr_cells = []
    for h, w in zip(headers, col_widths):
        hdr_cells.append(f" {h.center(w)} ")
    print("|" + "|".join(hdr_cells) + "|")
    print(header_sep)

    # Data rows
    for r in rows:
        cells = []
        for i, w in enumerate(col_widths):
            val = str(r[i]) if i < len(r) else ""
            # Left align text if it's long, right align if numeric
            clean_val = val.replace(".", "").replace("%", "").replace("ms", "").replace("scans", "").strip()
            if clean_val.isdigit():
                cells.append(f" {val.rjust(w)} ")
            else:
                cells.append(f" {val.ljust(w)} ")
        print("|" + "|".join(cells) + "|")

    print(bot_border)


# ==============================================================================
# 2. TRAINING CONFIGURATIONS
# ==============================================================================

FOV_CONFIG_DETAILS = {
    "Model Name": "FOV AI Scanner Classifier",
    "Target Task": "Field-of-View Classification (FULL_ARCH vs HEAD_ONLY)",
    "Backbone Architecture": "MobileNetV3-Small (Pretrained ImageNet-1K)",
    "Backbone Status": "Frozen Feature Extractor (Linear Probing)",
    "Classification Head": "Sequential(Dropout(p=0.4), Linear(1024 -> 2))",
    "Input Resolution": "224 x 224 x 3",
    "Preprocessing Pipeline": "Patient text removal + letterbox aspect-ratio preserving padding",
    "Data Augmentation": "RandomHorizontalFlip(p=0.5), RandomRotation(±8°), ColorJitter(0.2, 0.2)",
    "Normalization": "ImageNet Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]",
    "Loss Function": "Cross-Entropy Loss (torch.nn.CrossEntropyLoss)",
    "Optimizer": "AdamW (lr=1e-3, weight_decay=1e-2)",
    "Batch Size": "8",
    "Total Epochs": "15",
    "Train / Validation Split": "80% Train (12 scans) / 20% Val (4 scans)",
    "Model Weights Checkpoint": "fov_model.pt",
    "Default Dataset Directory": "data/fov_dataset (subfolders: FULL_ARCH, HEAD_ONLY)",
    "Evaluation Strategy": "Full cohort re-testing (identical training dataset due to limited data)",
}

SNN_CONFIG_DETAILS = {
    "Model Name": "SNN AI Bilateral Asymmetry Detector",
    "Target Task": "Bilateral Cerebral Hemisphere Asymmetry Detection (Ischemic Stroke)",
    "Backbone Architecture": "Multi-Scale Spatial Siamese ResNet-18 (Twin Encoder: Layer3 14x14 + Layer4 7x7)",
    "Comparison Head": "Multi-Scale Spatial Pooling: (Max + Avg) Diff L3 (512) ⊕ (Max + Avg) Diff L4 (1024) ⊕ Cosine Discrepancy (1) -> 1537 -> 256 -> LN -> GELU -> Dropout(0.2) -> 64 -> LN -> GELU -> Dropout(0.1) -> 1",
    "Training Strategy": "Two-Stage Transfer Learning (Warmup + Fine-Tuning)",
    "Stage 1 (Warmup)": "8 Epochs, Frozen Backbone, AdamW lr=1.5e-3 on Head",
    "Stage 2 (Fine-Tuning)": "14 Epochs, Unfrozen Layer4 + Head, Cosine Annealing, AdamW lr=2e-5 (layer4), 5e-4 (head)",
    "Input Resolution": "224 x 224 x 3 per hemisphere (Bilateral Pair)",
    "Preprocessing Pipeline": "Skull stripping, sagittal midline alignment, mirrored hemisphere extraction, disk caching",
    "Synthetic Augmentation": "Balanced 1:1 Dual-Mode: Focal arterial cutoff + Territorial hypodensity & Jittered bilateral healthy",
    "Data Augmentation": "Bilateral horizontal swap (p=0.5), Synchronized affine jitter (±4°, ±4px), Brightness/Contrast jitter",
    "Normalization": "ImageNet Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]",
    "Loss Function": "Binary Cross-Entropy Loss with Logits (BCEWithLogitsLoss)",
    "Batch Size": "8",
    "Total Epochs": "22 (8 Warmup + 14 Fine-Tuning)",
    "Train / Validation Split": "80% Train / 20% Val (Stratified Balanced Split)",
    "Model Weights Checkpoint": "best_model.pt",
    "Default Dataset Directory": "data/snn_dataset (subfolders: NORMAL, ABNORMAL) & data/test",
    "Evaluation Strategy": "Full cohort re-testing (identical training dataset due to limited data)",
}


def display_configuration_table(model_type: str):
    """Displays the training configuration table for the given model."""
    if model_type.upper() == "FOV":
        title = "TRAINING CONFIGURATION: FOV AI (MobileNetV3-Small)"
        data = FOV_CONFIG_DETAILS
    elif model_type.upper() == "SNN":
        title = "TRAINING CONFIGURATION: SNN AI (Siamese ResNet-18)"
        data = SNN_CONFIG_DETAILS
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    rows = [[k, v] for k, v in data.items()]
    print_box_table(title, ["Hyperparameter / Parameter", "Configuration Value"], rows, col_widths=[32, 54])


# ==============================================================================
# 3. METRICS COMPUTATION ENGINE
# ==============================================================================

def compute_binary_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[float],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of binary classification metrics.
    Assumes class 0 is Negative (e.g. FULL_ARCH or HEALTHY)
    and class 1 is Positive (e.g. HEAD_ONLY or STROKE).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true_arr)
    acc = accuracy_score(y_true_arr, y_pred_arr) * 100.0
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr) * 100.0

    # Sensitivity / Recall (Positive Class 1)
    sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    # Specificity (Negative Class 0)
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    # Precision / PPV (Positive Class 1)
    precision_pos = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    # Precision / NPV (Negative Class 0)
    npv = (tn / (tn + fn) * 100.0) if (tn + fn) > 0 else 0.0

    # Precision for negative class (TN / (TN + FN))
    precision_neg = npv
    recall_neg = specificity

    # F1-Scores
    f1_pos = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0) * 100.0
    f1_neg = f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0) * 100.0
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0) * 100.0
    f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0) * 100.0

    # ROC-AUC
    try:
        if len(np.unique(y_true_arr)) > 1:
            auc = roc_auc_score(y_true_arr, y_probs_arr) * 100.0
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    return {
        "total_samples": total,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_pos,
        "npv": npv,
        "f1_score": f1_pos,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": auc,
        "class_breakdown": {
            class_names[0]: {
                "support": int(tn + fp),
                "precision": precision_neg,
                "recall": recall_neg,
                "f1": f1_neg,
            },
            class_names[1]: {
                "support": int(tp + fn),
                "precision": precision_pos,
                "recall": sensitivity,
                "f1": f1_pos,
            },
        },
    }


def display_text_confusion_matrix(cm: List[List[int]], class_names: List[str]):
    """Prints a clear 2x2 text confusion matrix table."""
    c0, c1 = class_names[0], class_names[1]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp

    print(f"\nCONFUSION MATRIX:")
    print("+---------------------------+---------------------------------------+-----------+")
    print(f"|      ACTUAL \\ PREDICTED   |  {c0.center(16)} |  {c1.center(16)} |   TOTAL   |")
    print("+---------------------------+-------------------+-------------------+-----------+")
    print(f"|  {c0.ljust(24)} |  {str(tn).center(6)} ({tn/total*100:4.1f}%) |  {str(fp).center(6)} ({fp/total*100:4.1f}%) |  {str(tn+fp).center(7)}  |")
    print(f"|  {c1.ljust(24)} |  {str(fn).center(6)} ({fn/total*100:4.1f}%) |  {str(tp).center(6)} ({tp/total*100:4.1f}%) |  {str(fn+tp).center(7)}  |")
    print("+---------------------------+-------------------+-------------------+-----------+")
    print(f"|  {'PREDICTED TOTAL'.ljust(24)} |  {str(tn+fn).center(17)} |  {str(fp+tp).center(17)} |  {str(total).center(7)}  |")
    print("+---------------------------+-------------------+-------------------+-----------+")


def display_performance_metrics_table(model_name: str, metrics: Dict[str, Any], class_names: List[str]):
    """Displays formatted performance summary and per-class metrics tables."""
    # Summary Table
    auc_str = f"{metrics['roc_auc']:.2f}%" if not np.isnan(metrics['roc_auc']) else "N/A"
    summary_rows = [
        ["Total Evaluated Scans", f"{metrics['total_samples']} scans"],
        ["Overall Accuracy", f"{metrics['accuracy']:.2f}%"],
        ["Balanced Accuracy", f"{metrics['balanced_accuracy']:.2f}%"],
        [f"Sensitivity / Recall ({class_names[1]})", f"{metrics['sensitivity']:.2f}%"],
        [f"Specificity ({class_names[0]})", f"{metrics['specificity']:.2f}%"],
        [f"Precision / PPV ({class_names[1]})", f"{metrics['precision']:.2f}%"],
        [f"Negative Predictive Value (NPV)", f"{metrics['npv']:.2f}%"],
        [f"F1-Score ({class_names[1]})", f"{metrics['f1_score']:.2f}%"],
        ["Macro Average F1-Score", f"{metrics['f1_macro']:.2f}%"],
        ["Weighted Average F1-Score", f"{metrics['f1_weighted']:.2f}%"],
        ["ROC Area Under Curve (AUC)", auc_str],
        ["Confusion Breakdown", f"TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}"],
    ]
    if "avg_latency_ms" in metrics:
        summary_rows.append(["Average Inference Latency", f"{metrics['avg_latency_ms']:.2f} ms / scan"])

    print_box_table(f"PERFORMANCE METRICS SUMMARY: {model_name}", ["Metric Name", "Evaluated Score"], summary_rows, col_widths=[36, 42])

    # Per-class Table
    cls_rows = []
    for c_name in class_names:
        c_data = metrics["class_breakdown"][c_name]
        cls_rows.append([
            c_name,
            str(c_data["support"]),
            f"{c_data['precision']:.2f}%",
            f"{c_data['recall']:.2f}%",
            f"{c_data['f1']:.2f}%",
        ])

    print_box_table(
        f"PER-CLASS PERFORMANCE BREAKDOWN: {model_name}",
        ["Class Identifier", "Support", "Precision", "Recall", "F1-Score"],
        cls_rows,
        col_widths=[18, 10, 14, 14, 14]
    )


# ==============================================================================
# 4. FOV AI EVALUATION
# ==============================================================================

def evaluate_fov(
    data_dir: str = "data/fov_dataset",
    model_weights: str = "fov_model.pt",
    device: str = "auto"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates FOV AI model on scans located in data_dir (FULL_ARCH vs HEAD_ONLY).
    Returns (metrics, scan_details).
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    classes = ["FULL_ARCH", "HEAD_ONLY"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"[!] FOV weights '{model_weights}' not found. Please train FOV model first.")

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    file_records = []

    for cls_name, cls_idx in class_to_idx.items():
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_path):
            print(f"[!] Warning: FOV directory '{cls_path}' not found.")
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith(valid_exts):
                full_path = os.path.join(cls_path, fname)
                file_records.append((full_path, fname, cls_idx, cls_name))

    if not file_records:
        raise ValueError(f"[!] No valid scan images found in '{data_dir}'.")

    # Load model once for batch evaluation
    model = get_fov_model(num_classes=2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_weights, map_location=dev))
    model.to(dev).eval()

    normalize = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred = []
    y_probs = []
    scan_details = []
    latencies = []

    for full_path, fname, true_idx, true_name in file_records:
        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            print(f"[-] Could not read {fname}, skipping.")
            continue

        t0 = time.perf_counter()
        clean_square = preprocess_for_fov(img_bgr, target_size=(224, 224))
        rgb = cv2.cvtColor(clean_square, cv2.COLOR_BGR2RGB)
        tensor = normalize(rgb).unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(elapsed_ms)
        pred_name = classes[pred_idx]
        prob_pos = float(probs[1])  # Probability of HEAD_ONLY
        conf = float(probs[pred_idx])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_probs.append(prob_pos)

        is_correct = (true_idx == pred_idx)
        scan_details.append({
            "filename": fname,
            "path": full_path,
            "true_class": true_name,
            "pred_class": pred_name,
            "confidence": conf,
            "head_only_prob": prob_pos,
            "is_correct": is_correct,
            "latency_ms": elapsed_ms,
        })

    metrics = compute_binary_metrics(y_true, y_pred, y_probs, classes)
    metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0

    return metrics, scan_details


# ==============================================================================
# 5. SNN AI EVALUATION
# ==============================================================================

def evaluate_snn(
    data_dir: str = SNNConfig.DATA_DIR,
    healthy_dir: str = SNNConfig.HEALTHY_DIR,
    stroke_dir: str = SNNConfig.STROKE_DIR,
    model_weights: str = SNNConfig.MODEL_SAVE_PATH,
    threshold: float = 0.50,
    device: str = "auto",
    refresh_cache: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates SNN AI model on bilateral CTA scans (HEALTHY vs STROKE).
    Uses the same folder as training data (healthy & stroke directories).
    Returns (metrics, scan_details).
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    classes = ["HEALTHY", "STROKE"]

    model, dev = load_snn_model(model_weights, device=dev)
    if model is None:
        raise FileNotFoundError(f"[!] SNN model checkpoint '{model_weights}' not found. Please train SNN model first.")

    # 1. Discover Scans
    healthy_files, stroke_files = discover_scans(healthy_dir, stroke_dir, data_dir)
    if not healthy_files and not stroke_files:
        raise ValueError(f"[!] No CTA scans found in healthy_dir='{healthy_dir}', stroke_dir='{stroke_dir}', or data_dir='{data_dir}'.")

    # 2. Ingest Cohort (using preprocessed disk cache or live refresh)
    cohort = load_cohort_with_cache(healthy_files, stroke_files, target_size=SNNConfig.INPUT_SIZE, force_refresh=refresh_cache)
    if not cohort:
        raise ValueError("[!] Cohort preprocessing returned 0 samples.")

    y_true = []
    y_pred = []
    y_probs = []
    scan_details = []
    latencies = []

    # Fallback map for filenames
    all_files_ordered = [(f, 0.0, "HEALTHY") for f in healthy_files] + [(f, 1.0, "STROKE") for f in stroke_files]

    for idx, item in enumerate(cohort):
        left_tensor = item[0]
        right_tensor = item[1]
        true_label = item[2]
        img_path = item[3] if len(item) > 3 else (all_files_ordered[idx][0] if idx < len(all_files_ordered) else f"scan_{idx}")
        fname = os.path.basename(img_path)

        t0 = time.perf_counter()
        prob = predict_snn(left_tensor, right_tensor, model=model, device=dev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if prob is None:
            prob = 0.50

        pred_idx = 1 if prob >= threshold else 0
        true_idx = int(true_label)

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_probs.append(prob)
        latencies.append(elapsed_ms)

        is_correct = (true_idx == pred_idx)
        scan_details.append({
            "filename": fname,
            "path": img_path,
            "true_class": classes[true_idx],
            "pred_class": classes[pred_idx],
            "asymmetry_prob": prob,
            "decision_threshold": threshold,
            "is_correct": is_correct,
            "latency_ms": elapsed_ms,
        })

    metrics = compute_binary_metrics(y_true, y_pred, y_probs, classes)
    metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["decision_threshold"] = threshold

    return metrics, scan_details


# ==============================================================================
# 6. VISUALIZATION ENGINE (CONFUSION MATRICES)
# ==============================================================================

def plot_confusion_matrix_heatmap(
    cm: List[List[int]],
    class_names: List[str],
    title: str,
    subtitle: str,
    save_path: str,
    metrics: Dict[str, Any] = None,
    color_theme: str = "cyan"
):
    """
    Renders an aesthetically superior, publication-grade Confusion Matrix heatmap
    using a customized Light Mode theme aligned with clinical reporting standards.
    """
    cm_arr = np.array(cm)
    total = np.sum(cm_arr)
    cm_norm = cm_arr.astype("float") / total * 100.0

    # Color palettes (Light Mode)
    if color_theme == "cyan":
        cmap = plt.cm.Blues
        accent_color = "#0284c7"
    else:  # orange
        cmap = plt.cm.Oranges
        accent_color = "#ea580c"

    fig, ax = plt.subplots(figsize=(7.5, 6.8), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    # Plot image
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap, alpha=0.90)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#334155", labelsize=9)
    cbar.outline.set_edgecolor("#cbd5e1")
    cbar.set_label("Sample Count", color="#1e293b", fontsize=10, labelpad=8)

    # Ticks & Labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold", color="#0f172a")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="bold", color="#0f172a")

    ax.set_xlabel("Predicted Diagnosis Label", fontsize=12, fontweight="semibold", color="#1e293b", labelpad=10)
    ax.set_ylabel("True Ground Truth Label", fontsize=12, fontweight="semibold", color="#1e293b", labelpad=10)

    # Spines
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.2)

    # Text annotations in cells
    thresh = cm_arr.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm_arr[i, j]
            pct = cm_norm[i, j]
            txt_color = "#ffffff" if count > thresh else "#0f172a"

            cell_tag = ""
            if i == 0 and j == 0:
                cell_tag = "TN"
            elif i == 0 and j == 1:
                cell_tag = "FP"
            elif i == 1 and j == 0:
                cell_tag = "FN"
            elif i == 1 and j == 1:
                cell_tag = "TP"

            ax.text(
                j, i,
                f"[{cell_tag}]\n{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=txt_color,
                fontsize=13,
                fontweight="bold"
            )

    # Title & Subtitle
    plt.suptitle(title, fontsize=14, fontweight="bold", color="#0f172a", y=0.98)
    ax.set_title(subtitle, fontsize=10, color="#64748b", pad=12)

    # Metric box at bottom
    if metrics:
        acc = metrics.get("accuracy", 0.0)
        bal_acc = metrics.get("balanced_accuracy", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        sens = metrics.get("sensitivity", 0.0)
        spec = metrics.get("specificity", 0.0)

        stats_text = (
            f"Accuracy: {acc:.1f}%  |  Balanced Acc: {bal_acc:.1f}%  |  "
            f"Sensitivity: {sens:.1f}%  |  Specificity: {spec:.1f}%  |  F1: {f1:.1f}%"
        )
        fig.text(
            0.5, 0.02, stats_text,
            ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
            color="#0369a1"
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix visualization to: {save_path}")


def plot_combined_comparison(
    fov_metrics: Dict[str, Any],
    snn_metrics: Dict[str, Any],
    save_path: str
):
    """
    Creates a comprehensive dual comparison figure in Light Mode displaying:
      - Left: FOV AI Confusion Matrix
      - Right: SNN AI Confusion Matrix
      - Bottom: Side-by-side Comparative Performance Bar Chart
    """
    fig = plt.figure(figsize=(15, 8.5), dpi=300, facecolor="#ffffff")

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.35, wspace=0.25)
    ax_fov_cm = fig.add_subplot(gs[0, 0])
    ax_snn_cm = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[1, :])

    # 1. FOV Confusion Matrix Heatmap
    cm_fov = np.array(fov_metrics["confusion_matrix"])
    im1 = ax_fov_cm.imshow(cm_fov, cmap=plt.cm.Blues, alpha=0.90)
    ax_fov_cm.set_facecolor("#ffffff")
    ax_fov_cm.set_xticks([0, 1])
    ax_fov_cm.set_yticks([0, 1])
    ax_fov_cm.set_xticklabels(["FULL_ARCH", "HEAD_ONLY"], color="#0f172a", fontweight="bold")
    ax_fov_cm.set_yticklabels(["FULL_ARCH", "HEAD_ONLY"], color="#0f172a", fontweight="bold")
    ax_fov_cm.set_title("FOV AI (MobileNetV3-Small) Confusion Matrix", color="#0284c7", fontsize=11, pad=8)
    ax_fov_cm.set_xlabel("Predicted Label", color="#475569", fontsize=9.5)
    ax_fov_cm.set_ylabel("True Ground Truth", color="#475569", fontsize=9.5)

    for spine in ax_fov_cm.spines.values():
        spine.set_color("#cbd5e1")

    for i in range(2):
        for j in range(2):
            cnt = cm_fov[i, j]
            pct = cnt / np.sum(cm_fov) * 100.0
            tag = ["TN", "FP"] if i == 0 else ["FN", "TP"]
            ax_fov_cm.text(j, i, f"[{tag[j]}]\n{cnt}\n({pct:.1f}%)", ha="center", va="center", color="#ffffff" if cnt > cm_fov.max()/2 else "#0f172a", fontweight="bold")

    # 2. SNN Confusion Matrix Heatmap
    cm_snn = np.array(snn_metrics["confusion_matrix"])
    im2 = ax_snn_cm.imshow(cm_snn, cmap=plt.cm.Oranges, alpha=0.90)
    ax_snn_cm.set_facecolor("#ffffff")
    ax_snn_cm.set_xticks([0, 1])
    ax_snn_cm.set_yticks([0, 1])
    ax_snn_cm.set_xticklabels(["HEALTHY", "STROKE"], color="#0f172a", fontweight="bold")
    ax_snn_cm.set_yticklabels(["HEALTHY", "STROKE"], color="#0f172a", fontweight="bold")
    ax_snn_cm.set_title("SNN AI (Siamese ResNet-18) Confusion Matrix", color="#ea580c", fontsize=11, pad=8)
    ax_snn_cm.set_xlabel("Predicted Label", color="#475569", fontsize=9.5)
    ax_snn_cm.set_ylabel("True Ground Truth", color="#475569", fontsize=9.5)

    for spine in ax_snn_cm.spines.values():
        spine.set_color("#cbd5e1")

    for i in range(2):
        for j in range(2):
            cnt = cm_snn[i, j]
            pct = cnt / np.sum(cm_snn) * 100.0
            tag = ["TN", "FP"] if i == 0 else ["FN", "TP"]
            ax_snn_cm.text(j, i, f"[{tag[j]}]\n{cnt}\n({pct:.1f}%)", ha="center", va="center", color="#ffffff" if cnt > cm_snn.max()/2 else "#0f172a", fontweight="bold")

    # 3. Comparative Performance Bar Chart
    categories = ["Accuracy", "Balanced Acc", "Sensitivity", "Specificity", "F1-Score"]
    fov_vals = [
        fov_metrics["accuracy"],
        fov_metrics["balanced_accuracy"],
        fov_metrics["sensitivity"],
        fov_metrics["specificity"],
        fov_metrics["f1_score"],
    ]
    snn_vals = [
        snn_metrics["accuracy"],
        snn_metrics["balanced_accuracy"],
        snn_metrics["sensitivity"],
        snn_metrics["specificity"],
        snn_metrics["f1_score"],
    ]

    x = np.arange(len(categories))
    width = 0.32

    ax_bars.set_facecolor("#ffffff")
    rects1 = ax_bars.bar(x - width/2, fov_vals, width, label="FOV AI (MobileNetV3)", color="#0284c7", alpha=0.90)
    rects2 = ax_bars.bar(x + width/2, snn_vals, width, label="SNN AI (Siamese ResNet-18)", color="#ea580c", alpha=0.90)

    ax_bars.set_ylabel("Performance (%)", color="#1e293b", fontsize=10.5)
    ax_bars.set_title("Key Performance Benchmarks Comparison", color="#0f172a", fontsize=11.5, pad=8)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(categories, color="#0f172a", fontsize=10, fontweight="bold")
    ax_bars.set_ylim(0, 115)
    ax_bars.grid(axis="y", color="#e2e8f0", linestyle="--", alpha=0.8)
    ax_bars.legend(facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#0f172a", fontsize=9.5)

    for spine in ax_bars.spines.values():
        spine.set_color("#cbd5e1")

    # Add bar value labels
    for rect in rects1:
        h = rect.get_height()
        ax_bars.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         color="#0284c7", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax_bars.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         color="#ea580c", fontsize=8.5, fontweight="bold")

    plt.suptitle("ANTIGRAVITY STROKE AI - DUAL MODEL EVALUATION BENCHMARK", color="#0f172a", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
    plt.close()
    print(f"[+] Saved Dual Comparison Dashboard to: {save_path}")


# ==============================================================================
# 7. SCAN-BY-SCAN DETAILS TABLE
# ==============================================================================

def display_scan_details_table(model_name: str, scan_details: List[Dict[str, Any]], is_snn: bool = False):
    """Prints a detailed per-scan diagnostic concordance table."""
    headers = ["Scan File", "Ground Truth", "Predicted", "Confidence / Prob", "Status", "Latency"]
    rows = []
    for s in scan_details:
        status_str = "PASS" if s["is_correct"] else "MISS"
        if is_snn:
            prob_str = f"{s['asymmetry_prob']*100:.1f}%"
        else:
            prob_str = f"{s['confidence']*100:.1f}%"

        rows.append([
            s["filename"],
            s["true_class"],
            s["pred_class"],
            prob_str,
            status_str,
            f"{s['latency_ms']:.1f}ms",
        ])

    print_box_table(f"PER-SCAN INFERENCE AUDIT: {model_name}", headers, rows, col_widths=[14, 14, 14, 18, 8, 10])


# ==============================================================================
# 8. MARKDOWN & JSON REPORT GENERATION
# ==============================================================================

def save_markdown_report(
    fov_metrics: Dict[str, Any],
    snn_metrics: Dict[str, Any],
    report_file: str = "reports/evaluation_report.md"
):
    """Generates an extensive GitHub Markdown summary report."""
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    auc_fov = f"{fov_metrics['roc_auc']:.2f}%" if not np.isnan(fov_metrics['roc_auc']) else "N/A"
    auc_snn = f"{snn_metrics['roc_auc']:.2f}%" if not np.isnan(snn_metrics['roc_auc']) else "N/A"

    lines = [
        "# Antigravity Stroke AI - Model Evaluation Report",
        "",
        "> **Note:** Testing is performed on the training scan cohorts due to clinical dataset size constraints.",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Metric | FOV AI (MobileNetV3-Small) | SNN AI (Siamese ResNet-18) |",
        "| :--- | :--- | :--- |",
        f"| **Target Task** | Field-of-View Classification | Bilateral Asymmetry Detection |",
        f"| **Cohort Evaluated** | {fov_metrics['total_samples']} scans | {snn_metrics['total_samples']} scans |",
        f"| **Overall Accuracy** | **{fov_metrics['accuracy']:.2f}%** | **{snn_metrics['accuracy']:.2f}%** |",
        f"| **Balanced Accuracy** | **{fov_metrics['balanced_accuracy']:.2f}%** | **{snn_metrics['balanced_accuracy']:.2f}%** |",
        f"| **Sensitivity / Recall** | {fov_metrics['sensitivity']:.2f}% | {snn_metrics['sensitivity']:.2f}% |",
        f"| **Specificity** | {fov_metrics['specificity']:.2f}% | {snn_metrics['specificity']:.2f}% |",
        f"| **Precision (PPV)** | {fov_metrics['precision']:.2f}% | {snn_metrics['precision']:.2f}% |",
        f"| **Negative Pred Value (NPV)** | {fov_metrics['npv']:.2f}% | {snn_metrics['npv']:.2f}% |",
        f"| **F1-Score (Positive)** | {fov_metrics['f1_score']:.2f}% | {snn_metrics['f1_score']:.2f}% |",
        f"| **Macro Avg F1-Score** | {fov_metrics['f1_macro']:.2f}% | {snn_metrics['f1_macro']:.2f}% |",
        f"| **ROC-AUC** | {auc_fov} | {auc_snn} |",
        f"| **Average Latency** | {fov_metrics.get('avg_latency_ms', 0):.2f} ms | {snn_metrics.get('avg_latency_ms', 0):.2f} ms |",
        "",
        "## 2. Confusion Matrices",
        "",
        "### FOV AI Confusion Matrix (`FULL_ARCH` vs `HEAD_ONLY`)",
        f"- **True Negatives (`FULL_ARCH`):** {fov_metrics['tn']}",
        f"- **False Positives (`HEAD_ONLY` misclassification):** {fov_metrics['fp']}",
        f"- **False Negatives (`FULL_ARCH` misclassification):** {fov_metrics['fn']}",
        f"- **True Positives (`HEAD_ONLY`):** {fov_metrics['tp']}",
        "",
        "### SNN AI Confusion Matrix (`HEALTHY` vs `STROKE`)",
        f"- **True Negatives (`HEALTHY`):** {snn_metrics['tn']}",
        f"- **False Positives (`STROKE` false alarm):** {snn_metrics['fp']}",
        f"- **False Negatives (`STROKE` missed):** {snn_metrics['fn']}",
        f"- **True Positives (`STROKE` confirmed):** {snn_metrics['tp']}",
        "",
        "## 3. Training Configurations",
        "",
        "### FOV AI Architecture & Training Parameters",
        "| Parameter | Configuration |",
        "| :--- | :--- |",
    ]

    for k, v in FOV_CONFIG_DETAILS.items():
        lines.append(f"| **{k}** | {v} |")

    lines.extend([
        "",
        "### SNN AI Architecture & Training Parameters",
        "| Parameter | Configuration |",
        "| :--- | :--- |",
    ])

    for k, v in SNN_CONFIG_DETAILS.items():
        lines.append(f"| **{k}** | {v} |")

    lines.extend([
        "",
        "## 4. Visual Artifacts",
        "- FOV Confusion Matrix: `reports/confusion_matrix_fov.png`",
        "- SNN Confusion Matrix: `reports/confusion_matrix_snn.png`",
        "- Comparative Benchmark: `reports/confusion_matrix_comparison.png`",
    ])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[+] Saved Markdown summary report to: {report_file}")


# ==============================================================================
# 9. MAIN RUNNER & CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Evaluation Suite for FOV AI and SNN AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, choices=["all", "fov", "snn"], default="all",
                        help="Select model to evaluate ('all', 'fov', or 'snn')")
    parser.add_argument("--fov-dir", type=str, default="data/fov_dataset",
                        help="Directory containing FOV training/testing scans")
    parser.add_argument("--fov-weights", type=str, default="fov_model.pt",
                        help="Path to trained FOV checkpoint")
    parser.add_argument("--snn-dir", type=str, default=SNNConfig.DATA_DIR,
                        help="Base data directory for SNN scans")
    parser.add_argument("--snn-healthy-dir", type=str, default=SNNConfig.HEALTHY_DIR,
                        help="Dedicated healthy scans directory for SNN")
    parser.add_argument("--snn-stroke-dir", type=str, default=SNNConfig.STROKE_DIR,
                        help="Dedicated stroke scans directory for SNN")
    parser.add_argument("--cohort", type=str, choices=["reference", "test", "combined"], default="reference",
                        help="Select clinical cohort: 'reference' (data/snn_dataset), 'test' (data/test), or 'combined'")
    parser.add_argument("--snn-weights", type=str, default=SNNConfig.MODEL_SAVE_PATH,
                        help="Path to trained SNN checkpoint")
    parser.add_argument("--snn-threshold", type=float, default=0.50,
                        help="Decision probability threshold for SNN stroke classification")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Force recomputation of preprocessed tensors from raw scans instead of loading disk cache")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REPORTS_DIR,
                        help="Output directory for reports and confusion matrix plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating and saving PNG confusion matrix plots")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="Computation device")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Adjust directories if cohort flag is specified
    if args.cohort == "test":
        args.snn_healthy_dir = "data/test/normal"
        args.snn_stroke_dir = "data/test/abnormal"
    elif args.cohort == "reference":
        args.snn_healthy_dir = "data/snn_dataset/NORMAL" if os.path.isdir("data/snn_dataset/NORMAL") else args.snn_healthy_dir
        args.snn_stroke_dir = "data/snn_dataset/ABNORMAL" if os.path.isdir("data/snn_dataset/ABNORMAL") else args.snn_stroke_dir

    print(format_title_banner("ANTIGRAVITY STROKE AI - DUAL MODEL EVALUATION SUITE"))
    print(f" Execution Date & Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Target Evaluation     : {args.model.upper()}")
    print(f" Reports Output Path   : {os.path.abspath(args.output_dir)}")
    print(f" Selected Device       : {args.device.upper()}")
    print("=" * 88)

    fov_metrics = None
    snn_metrics = None
    fov_details = None
    snn_details = None

    # --------------------------------------------------------------------------
    # EVALUATE FOV AI
    # --------------------------------------------------------------------------
    if args.model in ("all", "fov"):
        print(format_title_banner("EVALUATING MODEL 1: FOV AI (Field-of-View Gate)"))
        display_configuration_table("FOV")

        try:
            fov_metrics, fov_details = evaluate_fov(
                data_dir=args.fov_dir,
                model_weights=args.fov_weights,
                device=args.device,
            )
            display_performance_metrics_table("FOV AI (MobileNetV3-Small)", fov_metrics, ["FULL_ARCH", "HEAD_ONLY"])
            display_text_confusion_matrix(fov_metrics["confusion_matrix"], ["FULL_ARCH", "HEAD_ONLY"])
            display_scan_details_table("FOV AI", fov_details, is_snn=False)

            if not args.no_plots:
                fov_cm_plot = os.path.join(args.output_dir, "confusion_matrix_fov.png")
                plot_confusion_matrix_heatmap(
                    cm=fov_metrics["confusion_matrix"],
                    class_names=["FULL_ARCH", "HEAD_ONLY"],
                    title="FOV AI Classifier - Confusion Matrix",
                    subtitle=f"MobileNetV3-Small | N={fov_metrics['total_samples']} Scans | Accuracy: {fov_metrics['accuracy']:.1f}%",
                    save_path=fov_cm_plot,
                    metrics=fov_metrics,
                    color_theme="cyan"
                )

        except Exception as e:
            print(f"[!] Error during FOV AI evaluation: {e}")
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------------------------
    # EVALUATE SNN AI
    # --------------------------------------------------------------------------
    if args.model in ("all", "snn"):
        print(format_title_banner("EVALUATING MODEL 2: SNN AI (Siamese Asymmetry Detector)"))
        display_configuration_table("SNN")

        try:
            snn_metrics, snn_details = evaluate_snn(
                data_dir=args.snn_dir,
                healthy_dir=args.snn_healthy_dir,
                stroke_dir=args.snn_stroke_dir,
                model_weights=args.snn_weights,
                threshold=args.snn_threshold,
                device=args.device,
                refresh_cache=args.refresh_cache,
            )
            display_performance_metrics_table("SNN AI (Siamese ResNet-18)", snn_metrics, ["HEALTHY", "STROKE"])
            display_text_confusion_matrix(snn_metrics["confusion_matrix"], ["HEALTHY", "STROKE"])
            display_scan_details_table("SNN AI", snn_details, is_snn=True)

            if not args.no_plots:
                snn_cm_plot = os.path.join(args.output_dir, "confusion_matrix_snn.png")
                plot_confusion_matrix_heatmap(
                    cm=snn_metrics["confusion_matrix"],
                    class_names=["HEALTHY", "STROKE"],
                    title="SNN AI Siamese Detector - Confusion Matrix",
                    subtitle=f"Siamese ResNet-18 | Threshold={args.snn_threshold:.2f} | N={snn_metrics['total_samples']} Scans",
                    save_path=snn_cm_plot,
                    metrics=snn_metrics,
                    color_theme="magenta"
                )

        except Exception as e:
            print(f"[!] Error during SNN AI evaluation: {e}")
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------------------------
    # COMBINED DASHBOARD & REPORT EXPORT
    # --------------------------------------------------------------------------
    if fov_metrics and snn_metrics:
        if not args.no_plots:
            comp_plot = os.path.join(args.output_dir, "confusion_matrix_comparison.png")
            plot_combined_comparison(fov_metrics, snn_metrics, comp_plot)

        # JSON Export
        json_report_path = os.path.join(args.output_dir, "evaluation_results.json")
        combined_payload = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {
                "fov_ai": {
                    "configuration": FOV_CONFIG_DETAILS,
                    "metrics": fov_metrics,
                    "scans": fov_details,
                },
                "snn_ai": {
                    "configuration": SNN_CONFIG_DETAILS,
                    "metrics": snn_metrics,
                    "scans": snn_details,
                }
            }
        }
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(combined_payload, f, indent=2)
        print(f"\n[+] Saved full structured JSON evaluation metrics to: {json_report_path}")

        # Markdown Export
        md_report_path = os.path.join(args.output_dir, "evaluation_report.md")
        save_markdown_report(fov_metrics, snn_metrics, md_report_path)

    print("\n" + "=" * 88)
    print("                     EVALUATION SUITE COMPLETED SUCCESSFULLY")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
