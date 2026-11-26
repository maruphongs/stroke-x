"""
Stroke-X: Core Headless Diagnostic Inference Engine
Decoupled API service for web dashboards, REST APIs, and automated pipelines.
"""

import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from analyze import calculate_asymmetry
from fov_ai import classify_fov
from preprocess import process_image
from snn_ai import Config as SNNConfig
from snn_ai import load_snn_model, predict_snn


def _to_json_safe(obj: Any) -> Any:
    """Recursively converts NumPy scalars/arrays to native Python JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


@dataclass
class StrokeAnalysisResult:
    success: bool
    error: Optional[str] = None
    execution_time_sec: float = 0.0

    # Diagnostic metrics
    fov_type: str = "UNKNOWN"
    fov_conf: float = 0.0
    snn_prob: float = 0.0
    snn_class: str = "UNKNOWN"
    affected_side: str = "NONE"
    verdict: str = ""
    sub_banner: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Detailed quantitative data
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    arc_profile: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Visualization images (BGR arrays)
    original_image: Optional[np.ndarray] = None
    aligned_image: Optional[np.ndarray] = None
    diagnostic_overlay: Optional[np.ndarray] = None
    stage_panels: List[np.ndarray] = field(default_factory=list)
    stage_titles: List[str] = field(default_factory=list)
    stage_subtitles: List[str] = field(default_factory=list)
    composite_dashboard: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable summary (excluding heavy numpy arrays)."""
        payload = {
            "success": self.success,
            "error": self.error,
            "execution_time_sec": round(float(self.execution_time_sec), 3),
            "fov": {
                "type": self.fov_type,
                "confidence": round(float(self.fov_conf), 4),
            },
            "radiological": {
                "snn_probability": round(float(self.snn_prob), 4),
                "snn_class": self.snn_class,
                "affected_side": self.affected_side,
                "verdict": self.verdict,
                "sub_banner": self.sub_banner,
                "bbox_count": len(self.bounding_boxes),
                "bounding_boxes": self.bounding_boxes,
            },
            "arc_profile": self.arc_profile,
            "recommendations": self.recommendations,
            "alignment": {
                "tilt_angle_deg": round(float(self.meta.get("angle", 0.0)), 2),
                "center_x": self.meta.get("cx"),
                "center_y": self.meta.get("cy"),
                "chest_cut_row": self.meta.get("chest_y"),
            },
        }
        return _to_json_safe(payload)


# Global model cache to avoid reloading weights per inference call
_CACHED_SNN_BUNDLE = None


def _download_file(url: str, dest_path: str) -> bool:
    """Downloads a file atomically from url to dest_path."""
    tmp_path = f"{dest_path}.download"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
        return True
    except Exception as e:
        print(f"[!] Failed to download {url}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def ensure_model_weights(
    snn_weights_path: str = SNNConfig.MODEL_SAVE_PATH,
    fov_weights_path: str = "fov_model.pt",
    snn_url: Optional[str] = None,
    fov_url: Optional[str] = None,
) -> Tuple[bool, bool]:
    """
    Ensures model checkpoints exist on disk, optionally downloading them from
    SNN_MODEL_URL / FOV_MODEL_URL (env vars or arguments) for cloud deployments.
    Returns (snn_available, fov_available).
    """
    snn_url = snn_url or os.getenv("SNN_MODEL_URL")
    fov_url = fov_url or os.getenv("FOV_MODEL_URL")

    if not os.path.exists(snn_weights_path) and snn_url:
        _download_file(snn_url, snn_weights_path)
    if not os.path.exists(fov_weights_path) and fov_url:
        _download_file(fov_url, fov_weights_path)

    return os.path.exists(snn_weights_path), os.path.exists(fov_weights_path)


def get_cached_snn_model(weights_path: str = SNNConfig.MODEL_SAVE_PATH):
    global _CACHED_SNN_BUNDLE
    if _CACHED_SNN_BUNDLE is None and os.path.exists(weights_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, device = load_snn_model(weights_path, device=device)
            if model is not None:
                _CACHED_SNN_BUNDLE = (model, device)
        except Exception as e:
            print(f"[!] Warning: Could not cache SNN model: {e}")
    return _CACHED_SNN_BUNDLE


def load_image_input(
    image_input: Union[str, bytes, np.ndarray],
) -> Optional[np.ndarray]:
    """Helper to convert various input types into a valid BGR numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            return None
        return cv2.imread(image_input)
    return None


def run_stroke_pipeline(
    image_input: Union[str, bytes, np.ndarray],
    case_name: str = "scan",
    snn_weights_path: str = "best_model.pt",
    fov_weights_path: str = "fov_model.pt",
    asym_threshold: float = 50.0,
    bbox_threshold: float = 85.0,
    save_dashboard_to_disk: bool = False,
    generate_composite: bool = False,
    output_dir: str = "reports",
) -> StrokeAnalysisResult:
    """
    Executes the complete end-to-end Stroke-X diagnostic pipeline headlessly.
    Returns a structured StrokeAnalysisResult object containing metrics, profiles, and visual boards.
    """
    start_time = time.time()

    # 1. Image Ingestion
    img_bgr = load_image_input(image_input)
    if img_bgr is None:
        return StrokeAnalysisResult(
            success=False,
            error="Could not decode or load input image.",
            execution_time_sec=time.time() - start_time,
        )

    # 2. FOV Classification Gatekeeper
    fov_type = "UNKNOWN"
    fov_conf = 1.0
    if os.path.exists(fov_weights_path):
        try:
            fov_type, fov_conf = classify_fov(img_bgr, model_weights=fov_weights_path)
        except Exception as e:
            print(f"[!] FOV classification skipped: {e}")

    # 3. Morphological Alignment & Midline Straightening (Strictly Headless)
    try:
        prep_res = process_image(
            img_bgr,
            target_size=SNNConfig.INPUT_SIZE,
            mode="auto",
            interactive_fallback=False,  # Headless: never prompt GUI
            fov_type=fov_type if fov_type != "UNKNOWN" else None,
            preview=False,
        )
    except Exception as e:
        return StrokeAnalysisResult(
            success=False,
            error=f"Preprocessing failed: {e}",
            execution_time_sec=time.time() - start_time,
        )

    if prep_res is None or prep_res == "QUIT":
        return StrokeAnalysisResult(
            success=False,
            error="Midline alignment failed or scan dimensions too small.",
            execution_time_sec=time.time() - start_time,
        )

    left_tensor, right_tensor, left_high_res, right_high_res, original_aligned = (
        prep_res
    )
    meta = getattr(prep_res, "meta", {})

    # 4. SNN Inference (ResNet-18 Hemispheric Contrastive Model)
    snn_prob = 0.0
    if os.path.exists(snn_weights_path):
        try:
            cached = get_cached_snn_model(snn_weights_path)
            if cached is not None:
                snn_model, snn_device = cached
                prob = predict_snn(left_tensor, right_tensor, model=snn_model, device=snn_device)
            else:
                prob = predict_snn(left_tensor, right_tensor, weights_path=snn_weights_path)
            snn_prob = float(prob) if prob is not None else 0.0
        except Exception as e:
            print(f"[!] SNN prediction failed: {e}")

    snn_class = "STROKE" if snn_prob >= (asym_threshold / 100.0) else "HEALTHY"

    # 5. Morphological Spatial & Vessel Asymmetry Analysis
    final_overlay, debug_dash = calculate_asymmetry(
        original_aligned,
        min_asym_thresh=asym_threshold,
        bbox_thresh=bbox_threshold,
        meta=meta,
    )

    # 6. Quantitative Arc-Length Deficit & Hemispheric Localization
    affected_side = "NONE"
    profile_data = []
    profile = getattr(debug_dash, "profile", None)
    if profile:
        profile_data = profile
        if snn_class == "STROKE":
            tot_l = sum(row.get("left_px", 0) for row in profile)
            tot_r = sum(row.get("right_px", 0) for row in profile)
            tot_all = tot_l + tot_r
            ai = (tot_r - tot_l) / float(max(1, tot_all))
            if ai < -0.04:
                affected_side = "RIGHT"
            elif ai > 0.04:
                affected_side = "LEFT"
            else:
                affected_side = "BILATERAL"

    # 7. Formulate Clinical Verdict & Clinical Guidance
    has_deficit = snn_class == "STROKE"
    if has_deficit:
        if affected_side in ("LEFT", "RIGHT"):
            verdict = f"ASYMMETRY DETECTED: {affected_side} HEMISPHERE"
        elif affected_side == "BILATERAL":
            verdict = "ASYMMETRY DETECTED: BILATERAL DEFICIT"
        else:
            verdict = "ASYMMETRY DETECTED: DIFFUSE CEREBRAL DEFICIT"
    else:
        verdict = "SYMMETRICAL SCAN: NEGATIVE FOR SIGNIFICANT DEFICIT"

    sub_banner = f"Deficit Side: {affected_side} | SNN Prediction: {snn_class} ({snn_prob * 100:.1f}% asym)"

    # Clinical recommendations
    bounding_boxes = getattr(debug_dash, "bounding_boxes", []) or []
    n_boxes = len(bounding_boxes)
    if has_deficit:
        recommendations = [
            f"1. Significant unilateral vascular disparity identified in the {affected_side} hemisphere.",
            f"2. {n_boxes} high-deficit region(s) detected meeting contralateral void criteria (threshold >= {bbox_threshold}%).",
            "3. Recommend urgent clinical neurological examination (NIHSS) and acute stroke team activation.",
            "4. Consider emergency CT Perfusion (CTP) / MRI DWI to evaluate ischemic core vs. penumbra.",
        ]
    else:
        recommendations = [
            "1. CTA scan demonstrates bilateral intracranial arterial arborization symmetry.",
            "2. Negative for focal Large Vessel Occlusion (LVO) asymmetry in anterior/middle circulation.",
            "3. If clinical symptoms persist, evaluate for posterior circulation or lacunar small vessel infarction.",
            "4. Follow-up imaging (MRI Brain with DWI) suggested if focal clinical deficit persists.",
        ]

    # 8. Render Composite Infographic Dashboard (only if requested)
    composite_canvas = None
    if save_dashboard_to_disk or generate_composite:
        try:
            from report import generate_dashboard_image

            findings_dict = {
                "affected_side": affected_side,
                "snn_prob": snn_prob,
                "snn_class": snn_class,
                "bbox_count": n_boxes,
                "bounding_boxes": bounding_boxes,
            }
            composite_canvas, _ = generate_dashboard_image(
                image_path=case_name,
                final_image=final_overlay,
                debug_dash=debug_dash,
                triage={"skipped": True},
                verdict=verdict,
                sub_banner=sub_banner,
                findings=findings_dict,
                meta=meta,
                output_dir=output_dir if save_dashboard_to_disk else "reports",
                original_aligned=original_aligned,
            )
        except Exception as e:
            print(f"[!] Composite dashboard rendering error: {e}")

    # Panels from algorithmic pipeline
    panels = getattr(debug_dash, "panels", [])
    panel_titles = getattr(debug_dash, "panel_titles", [])
    panel_subtitles = getattr(debug_dash, "panel_subtitles", [])

    return StrokeAnalysisResult(
        success=True,
        error=None,
        execution_time_sec=time.time() - start_time,
        fov_type=fov_type,
        fov_conf=fov_conf,
        snn_prob=snn_prob,
        snn_class=snn_class,
        affected_side=affected_side,
        verdict=verdict,
        sub_banner=sub_banner,
        recommendations=recommendations,
        bounding_boxes=bounding_boxes,
        arc_profile=profile_data,
        meta=meta,
        original_image=img_bgr,
        aligned_image=original_aligned,
        diagnostic_overlay=final_overlay,
        stage_panels=panels,
        stage_titles=panel_titles,
        stage_subtitles=panel_subtitles,
        composite_dashboard=composite_canvas,
    )
