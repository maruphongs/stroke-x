"""
================================================================================
ANTIGRAVITY STROKE AI - TRAINING PARAMETERS & HYPERPARAMETERS COMPARISON
================================================================================
Generates publication-ready comparative tables contrasting the exact
TRAINING PARAMETERS and HYPERPARAMETERS between:
  1. Baseline Old SNN (.old/best_model.pt)
  2. Proposed Multi-Scale Spatial SNN (best_model.pt)

Outputs organized in:
  - reports/snn_comparison/tables/training_parameters_table.md
  - reports/snn_comparison/tables/training_parameters_table.tex
  - reports/snn_comparison/figures/training_parameters_table.png
  (and mirrored to parameter_comparison_table.* for backward compatibility)
================================================================================
"""

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
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_training_parameters():
    """
    Returns the comprehensive, clean comparison of training parameters & hyperparameters.
    """
    params = [
        # --- Core Training Setup ---
        ("Training Strategy / Paradigm", "Single-Stage End-to-End", "Two-Stage Transfer Learning (Warmup + Fine-Tuning)"),
        ("Total Training Epochs", "20 Epochs", "22 Epochs (8 Warmup + 14 Fine-Tuning)"),
        ("Batch Size", "8 pairs per batch", "8 pairs per batch"),
        ("Optimization Algorithm", "Adam", "AdamW (Decoupled Weight Decay)"),
        ("Initial Learning Rate (Classifier)", "1.0e-3 (Constant)", "Stage 1: 1.5e-3 | Stage 2: 5.0e-4"),
        ("Backbone Feature Extractor LR", "1.0e-3 (Constant)", "Stage 1: 0.0 (Frozen) | Stage 2: 2.0e-5 (Layer 4)"),
        ("Learning Rate Schedule", "None (Constant LR)", "Cosine Annealing (down to eta_min = 1.0e-6 in Stage 2)"),
        ("Weight Decay (L2 Reg.)", "0.0 (None)", "1.0e-3"),
        ("Loss Objective Function", "Binary Cross-Entropy (BCEWithLogitsLoss)", "Balanced Focal Loss (alpha=0.50, gamma=2.0)"),
        
        # --- Transfer Learning & Freezing ---
        ("Backbone Pretraining", "ImageNet-1K (ResNet-18)", "ImageNet-1K (ResNet-18)"),
        ("Backbone Freezing Protocol", "None (All layers trained end-to-end)", "Stage 1: Full Backbone Frozen (Conv1-L4)\nStage 2: Layer 4 Unfrozen, Conv1-L3 Frozen"),
        ("Active Trainable Parameters", "11,455,937 (100% from epoch 1)", "Stage 1: 410,881 (Classifier only, 3.5%)\nStage 2: 8,711,681 (Layer 4 + Classifier, 75.2%)"),
        
        # --- Data & Synthetic Augmentation ---
        ("Synthetic Stroke Generation", "None (Raw bilateral pairs only)", "1:1 Balanced Dual-Mode:\n- Focal arterial cutoff\n- Territorial hemispheric hypodensity"),
        ("Synthetic Multiplier", "1x (Raw scans only)", "6x Synthetic pairs per scan"),
        ("Bilateral Symmetry Augmentation", "Horizontal mirror only (p=0.5)", "Bilateral hemisphere swap (p=0.5)"),
        ("Spatial Geometric Jitter", "None", "Synchronized affine jitter:\n- Rotation: +/- 4 deg\n- Translation: +/- 4 px\n- Scale: 0.97 - 1.03"),
        ("Photometric Jitter", "None", "Synchronized Brightness (+/- 8%) & Contrast (+/- 8%)"),
        
        # --- Regularization & Normalization ---
        ("Dropout Regularization", "Dropout(0.30) -> Dropout(0.20)", "Dropout(0.20) -> Dropout(0.10)"),
        ("Feature Normalization", "Batch Normalization (BatchNorm1d)", "Layer Normalization (LayerNorm - Batch-Invariant)"),
        ("Input Tensor Resolution", "224 x 224 x 3 per hemisphere", "224 x 224 x 3 per hemisphere"),
        ("Train / Validation Split", "80% Train / 20% Holdout (Random)", "80% Train / 20% Holdout (Stratified Balanced)"),
    ]
    return params


def generate_markdown_table(params):
    lines = [
        "# Training Parameters & Hyperparameters Comparison",
        "",
        "> **Table:** Comprehensive comparison of training protocol, hyperparameters, optimization, and data augmentation between Conventional SNN and Proposed SNN.",
        "",
        "| Training Parameter / Hyperparameter | Conventional SNN | Proposed SNN |",
        "| :--- | :--- | :--- |",
    ]
    for param, old_val, new_val in params:
        clean_old = old_val.replace("\n", "<br>")
        clean_new = new_val.replace("\n", "<br>")
        lines.append(f"| **{param}** | {clean_old} | {clean_new} |")
    return "\n".join(lines)


def generate_latex_table(params):
    lines = [
        "% Academic Article Training Parameters Table (LaTeX Format)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comparison of Training Parameters, Optimization Hyperparameters, and Augmentation Strategies between Conventional SNN and Proposed SNN}",
        "\\label{tab:training_parameters_comparison}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lll}",
        "\\hline",
        "\\textbf{Training Parameter / Hyperparameter} & \\textbf{Conventional SNN} & \\textbf{Proposed SNN} \\\\",
        "\\hline",
    ]
    for param, old_val, new_val in params:
        clean_param = param.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
        clean_old = old_val.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("->", "$\\rightarrow$").replace("\n", " / ")
        clean_new = new_val.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("->", "$\\rightarrow$").replace("\n", " / ").replace("alpha", "$\\alpha$").replace("gamma", "$\\gamma$")
        lines.append(f"{clean_param} & {clean_old} & {clean_new} \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def render_table_graphic(params, save_path):
    fig, ax = plt.subplots(figsize=(15.0, 10.5), dpi=300, facecolor="#ffffff")
    ax.axis("off")
    ax.axis("tight")

    columns = [
        "Training Parameter / Hyperparameter",
        "Conventional SNN",
        "Proposed SNN",
    ]
    cell_data = [[param, old_v, new_v] for param, old_v, new_v in params]

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colWidths=[0.30, 0.35, 0.35],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.40)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
            cell.set_facecolor("#1e293b")  # Dark slate header
            cell.set_height(0.045)
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("#ffffff")
                
            if col == 0:
                cell.set_text_props(weight="semibold", color="#0f172a")
            elif col == 1:
                cell.set_text_props(color="#475569")
            else:
                cell.set_text_props(weight="medium", color="#0369a1")

    plt.title(
        "Quantitative Comparison of Model Training Parameters, Optimization, & Augmentation",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        pad=16,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved Training Parameters Table Image: {save_path}")


def run():
    print("=" * 80)
    print("  SNN COMPARISON: TRAINING PARAMETERS & HYPERPARAMETERS BENCHMARK")
    print("=" * 80)

    params = get_training_parameters()

    # 1. Output Markdown (both names for clarity and compatibility)
    md_content = generate_markdown_table(params)
    for fname in ["training_parameters_table.md", "parameter_comparison_table.md"]:
        md_path = os.path.join(TABLES_DIR, fname)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[+] Saved Markdown table: {md_path}")

    # 2. Output LaTeX
    tex_content = generate_latex_table(params)
    for fname in ["training_parameters_table.tex", "parameter_comparison_table.tex"]:
        tex_path = os.path.join(TABLES_DIR, fname)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        print(f"[+] Saved LaTeX table: {tex_path}")

    # 3. Render High-Res Graphic
    for fname in ["training_parameters_table.png", "parameter_comparison_table.png"]:
        img_path = os.path.join(FIGURES_DIR, fname)
        render_table_graphic(params, save_path=img_path)

    print("\n" + md_content)
    print("\n" + "=" * 80)
    print("Training parameters table generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    run()
