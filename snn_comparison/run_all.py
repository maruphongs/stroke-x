"""
================================================================================
SNN COMPARISON MASTER RUNNER - ANTIGRAVITY STROKE AI
================================================================================
Executes the complete SNN comparison pipeline in order:
  1. Architectural & Parameter Comparison (Markdown, LaTeX, Figure)
  2. Dual-Model Training Trajectory Benchmark (Curves, PDF, JSON)
  3. Multi-Cohort Clinical Evaluation Suite (Confusion Matrix, ROC, Report)

All artifacts are saved to structured folders under:
  reports/snn_comparison/
    ├── figures/
    ├── tables/
    ├── metrics/
    └── academic_article_evaluation_report.md
================================================================================
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from snn_comparison.compare_parameters import run as run_parameters
from snn_comparison.compare_training import run as run_training
from snn_comparison.evaluate_comparison import run as run_evaluation
from snn_comparison.generate_test_accuracy_table import run as run_test_accuracy


def main():
    print("#" * 85)
    print("      ANTIGRAVITY STROKE AI - COMPLETE SNN COMPARISON & BENCHMARK SUITE")
    print("#" * 85)
    print(f" Target Output Directory : {os.path.join(_PROJECT_ROOT, 'reports', 'snn_comparison')}")
    print("#" * 85 + "\n")

    # Step 1: Parameter & Architecture Comparison
    print(">>> [STEP 1/4] Generating Training Parameters & Hyperparameters Table ...")
    run_parameters()
    print("\n" + "-" * 85 + "\n")

    # Step 2: Training Trajectory Curves
    print(">>> [STEP 2/4] Generating Training Dynamics & Convergence Curves ...")
    run_training()
    print("\n" + "-" * 85 + "\n")

    # Step 3: Comparative Evaluation
    print(">>> [STEP 3/4] Running Dual-Model Multi-Cohort Clinical Evaluation ...")
    run_evaluation()
    print("\n" + "-" * 85 + "\n")

    # Step 4: Dedicated Test Accuracy Tables & Visuals
    print(">>> [STEP 4/4] Generating Dedicated Test Data Accuracy Evaluation Tables ...")
    run_test_accuracy()
    print("\n" + "#" * 85)
    print("ALL SNN COMPARISON ARTIFACTS GENERATED SUCCESSFULLY!")
    print(f"Organized output path: {os.path.join(_PROJECT_ROOT, 'reports', 'snn_comparison')}")
    print("#" * 85)


if __name__ == "__main__":
    main()
