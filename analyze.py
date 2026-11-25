import math

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

# --- Configuration & Hyperparameters ---
NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
ENDPOINT_NEIGHBOR_SUM = 11
MIN_BRANCH_LENGTH = 15

TRIM_FRACTION = 0.15     # Discard unmatched stroke regions when solving tx
Y_MATCH_TOLERANCE = 4    # Maximum vertical pixel difference to pair vessel points

# Deficit & High-Asymmetry Bounding Box Constants
MIN_DEFICIT_ASYM_THRESHOLD = 50.0   # Raised from 40.0: minimum % asymmetry to display color deficit
HIGH_DEFICIT_THRESHOLD = 85.0       # Asymmetry % threshold to trigger diagnostic bounding box ("when it really high")
MIN_BBOX_AREA = 1200                # Minimum pixel area required to draw a high-deficit bounding box

LAST_ASYMMETRY_PROFILE = None
LAST_BOUNDING_BOXES = None

class DashboardImage(np.ndarray):
    # Subclass of np.ndarray that carries the quantitative asymmetry profile, bounding boxes, and pipeline panels.
    def __new__(cls, input_array, profile=None, bounding_boxes=None, panels=None, panel_titles=None, panel_subtitles=None, panel_descriptions=None):
        obj = np.asarray(input_array).view(cls)
        obj.profile = profile
        obj.bounding_boxes = bounding_boxes
        obj.panels = panels if panels is not None else []
        obj.panel_titles = panel_titles if panel_titles is not None else []
        obj.panel_subtitles = panel_subtitles if panel_subtitles is not None else []
        obj.panel_descriptions = panel_descriptions if panel_descriptions is not None else []
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.profile = getattr(obj, "profile", None)
        self.bounding_boxes = getattr(obj, "bounding_boxes", None)
        self.panels = getattr(obj, "panels", None)
        self.panel_titles = getattr(obj, "panel_titles", None)
        self.panel_subtitles = getattr(obj, "panel_subtitles", None)
        self.panel_descriptions = getattr(obj, "panel_descriptions", None)


def merge_boxes_compact(boxes, iou_thresh=0.10, dist_thresh=6):
    # Merges overlapping or immediately adjacent bounding boxes across multi-scale detections.
    # boxes: list of (x, y, w, h, asym, area)
    if not boxes:
        return []
    
    # Sort primarily by area * (asym^0.6) descending
    boxes = sorted(boxes, key=lambda b: (b[4] * (b[5] ** 0.6)), reverse=True)
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1, asym1, area1 = boxes[i]
        cur_x1, cur_y1, cur_x2, cur_y2 = x1, y1, x1 + w1, y1 + h1
        max_asym = asym1
        tot_area = area1
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2, asym2, area2 = boxes[j]
                dx = max(0, max(cur_x1, x2) - min(cur_x2, x2 + w2))
                dy = max(0, max(cur_y1, y2) - min(cur_y2, y2 + h2))
                dist = np.hypot(dx, dy)
                ix1 = max(cur_x1, x2)
                iy1 = max(cur_y1, y2)
                ix2 = min(cur_x2, x2 + w2)
                iy2 = min(cur_y2, y2 + h2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                area_cur = (cur_x2 - cur_x1) * (cur_y2 - cur_y1)
                area_j = w2 * h2
                iou = inter / float(area_cur + area_j - inter) if (area_cur + area_j - inter) > 0 else 0
                overlap_ratio = inter / float(min(area_cur, area_j)) if min(area_cur, area_j) > 0 else 0
                if iou > iou_thresh or overlap_ratio > 0.25 or dist <= dist_thresh:
                    cur_x1 = min(cur_x1, x2)
                    cur_y1 = min(cur_y1, y2)
                    cur_x2 = max(cur_x2, x2 + w2)
                    cur_y2 = max(cur_y2, y2 + h2)
                    max_asym = max(max_asym, asym2)
                    tot_area += area2
                    used[j] = True
                    changed = True
        merged.append((cur_x1, cur_y1, cur_x2 - cur_x1, cur_y2 - cur_y1, max_asym, tot_area))
    return merged


def draw_simple_thin_box(img, x1, y1, x2, y2, color=(0, 0, 255), thickness=1):
    # Draws a simple, thin (1px) bounding box around a high-deficit region.
    # No percentage text, no badges, keeping the vessel view clean and unobstructed.

    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def logistic_growth_factor(x, x0=70.0, k=0.15, x_min=50.0, x_max=95.0):
    # Sigmoidal / Logistic growth function mapping asymmetry [x_min, x_max] to [0.0, 1.0].
    # Provides:
    #   1. Gentle baseline suppression for borderline/mild deficits (50-65%)
    #   2. Accelerated growth through the transition zone (65-80%)
    #   3. Smooth asymptotic saturation towards 1.0 for severe/critical deficits (>85%)

    z = k * (x - x0)
    raw = 1.0 / (1.0 + np.exp(-z))
    raw_min = 1.0 / (1.0 + np.exp(-k * (x_min - x0)))
    raw_max = 1.0 / (1.0 + np.exp(-k * (x_max - x0)))
    return float(np.clip((raw - raw_min) / max(raw_max - raw_min, 1e-6), 0.0, 1.0))

# --- 1. Vessel Segmentation & Skeletonization ---

def get_vessel_analysis(gray):
    # Extracts multi-scale vascular structures using CLAHE and dual Top-Hat morphology,
    # followed by morphological closing and area-based noise scrubbing.

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    
    # Dual-scale top-hat for fine and major vessels
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m1 = cv2.threshold(cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, k1), 15, 255, cv2.THRESH_BINARY)[1]
    
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    m2 = cv2.threshold(cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, k2), 15, 255, cv2.THRESH_BINARY)[1]
    
    combined = cv2.bitwise_or(m1, m2)
    
    # Scrub small noise islands < 200 pixels
    num, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    clean = np.zeros_like(combined)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > 200:
            clean[labels == i] = 255
            
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_k)
    return smoothed

def prune_skeleton_spurs(skeleton_img, min_branch_length=MIN_BRANCH_LENGTH):
    # Prunes noisy dead-end spurs caused by segmentation edge irregularities.
    skel = (skeleton_img > 0).astype(np.uint8)
    pruned_any = True

    while pruned_any:
        pruned_any = False
        neighbor_map = convolve(skel, NEIGHBOR_KERNEL, mode="constant", cval=0)
        endpoints = np.argwhere(neighbor_map == ENDPOINT_NEIGHBOR_SUM)

        for y, x in endpoints:
            y, x = int(y), int(x)
            if skel[y, x] == 0:
                continue

            branch_pixels = [(y, x)]
            visited = {(y, x)}
            curr_y, curr_x = y, x

            while len(branch_pixels) < min_branch_length:
                y0, y1 = max(0, curr_y - 1), min(skel.shape[0], curr_y + 2)
                x0, x1 = max(0, curr_x - 1), min(skel.shape[1], curr_x + 2)

                neighborhood = skel[y0:y1, x0:x1]
                neighbors = np.argwhere(neighborhood == 1)
                global_neighbors = [
                    (int(n[0]) + y0, int(n[1]) + x0)
                    for n in neighbors
                    if (int(n[0]) + y0, int(n[1]) + x0) != (curr_y, curr_x)
                ]
                unvisited = [n for n in global_neighbors if n not in visited]

                if len(unvisited) == 1:
                    branch_pixels.append(unvisited[0])
                    visited.add(unvisited[0])
                    curr_y, curr_x = unvisited[0]
                else:
                    break

            if len(branch_pixels) < min_branch_length:
                for py, px in branch_pixels:
                    skel[py, px] = 0
                pruned_any = True

    return skel * 255

# --- 2. 1D Horizontal Registration (Y-Axis Strictly Locked) ---

def compute_horizontal_registration(src_pts, tgt_pts, y_tol=Y_MATCH_TOLERANCE, trim_frac=TRIM_FRACTION):
    # Finds optimal horizontal translation (tx) while freezing the Y-axis.
    # Ensures y' = y exactly, preventing slice-level distortion.

    if len(src_pts) < 10 or len(tgt_pts) < 10:
        return None, None

    tree = KDTree(tgt_pts)
    tx = 0.0

    # Iteratively refine tx while pairing points only at matching Y levels
    for _ in range(15):
        cur_src = src_pts.copy()
        cur_src[:, 0] += tx

        _distances, indices = tree.query(cur_src)
        matched_tgt = tgt_pts[indices]

        # Only pair points lying on the exact same anatomical height (same horizontal slice)
        y_diff = np.abs(matched_tgt[:, 1] - cur_src[:, 1])
        valid = y_diff <= y_tol

        if np.sum(valid) < 5:
            break

        # Calculate horizontal offsets between paired points
        offsets_x = matched_tgt[valid, 0] - src_pts[valid, 0]

        # Trim outlier percentiles (discards occluded branches/asymmetries)
        q_low = np.percentile(offsets_x, trim_frac * 100.0)
        q_high = np.percentile(offsets_x, (1.0 - trim_frac) * 100.0)
        trimmed = offsets_x[(offsets_x >= q_low) & (offsets_x <= q_high)]

        if len(trimmed) == 0:
            break

        delta_tx = float(np.median(trimmed)) - tx
        tx += delta_tx
        if abs(delta_tx) < 1e-3:
            break

    # Strictly lock the Y row: y' = 0*x + 1*y + 0 = y
    M = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    aligned_src = src_pts.copy()
    aligned_src[:, 0] += tx

    return M, aligned_src

# --- 3. Bidirectional Asymmetry Pipeline ---

def draw_panel_card_header(panel_img, step_num: int, title: str, subtitle: str = ""):
    # Draws a clean medical card header and bottom status bar for an architecture step panel in Light Mode.
    h, w = panel_img.shape[:2]
    # Header bar
    header_h = 32
    overlay = panel_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.88, panel_img, 0.12, 0, panel_img)
    cv2.line(panel_img, (0, header_h), (w, header_h), (210, 215, 225), 1)

    # Step badge (e.g. S1, S2, etc.)
    badge_w = 46
    cv2.rectangle(panel_img, (6, 5), (6 + badge_w, header_h - 5), (200, 100, 0), -1)
    cv2.putText(panel_img, f"S{step_num}", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Title text
    cv2.putText(panel_img, title, (6 + badge_w + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 41, 59), 1, cv2.LINE_AA)

    # Bottom status bar if subtitle provided
    if subtitle:
        bot_h = 24
        overlay = panel_img.copy()
        cv2.rectangle(overlay, (0, h - bot_h), (w, h), (250, 252, 255), -1)
        cv2.addWeighted(overlay, 0.88, panel_img, 0.12, 0, panel_img)
        cv2.line(panel_img, (0, h - bot_h), (w, h - bot_h), (210, 215, 225), 1)
        cv2.putText(panel_img, subtitle, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (71, 85, 105), 1, cv2.LINE_AA)


def calculate_asymmetry(
    aligned_image,
    *args,
    min_asym_thresh=MIN_DEFICIT_ASYM_THRESHOLD,
    bbox_thresh=HIGH_DEFICIT_THRESHOLD,
    min_bbox_area=MIN_BBOX_AREA,
    meta=None,
    midline_x=None,
    **kwargs,
):
    # Backward compatibility for legacy positional calling:
    # calculate_asymmetry(left_gray, right_gray, original_image, ...)
    if len(args) >= 2 and isinstance(args[1], np.ndarray):
        aligned_image = args[1]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        aligned_image = args[0]

    if len(aligned_image.shape) == 3:
        full_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        output_img = aligned_image.copy()
    else:
        full_gray = aligned_image.copy()
        output_img = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)

    h, w = full_gray.shape
    mid = w // 2

    # 1. Vessel segmentation & skeleton extraction
    full_mask = get_vessel_analysis(full_gray)
    raw_skel = skeletonize(full_mask > 0).astype(np.uint8) * 255
    clean_skel = prune_skeleton_spurs(raw_skel, MIN_BRANCH_LENGTH)

    # 2. Extract skeleton coordinates for Left and Right hemispheres
    y_l, x_l = np.where(clean_skel[:, :mid] > 0)
    y_r, x_r = np.where(clean_skel[:, mid:] > 0)
    x_r = x_r + mid  # Global coordinates

    pts_left = np.column_stack((x_l, y_l)).astype(np.float32)
    pts_right = np.column_stack((x_r, y_r)).astype(np.float32)

    # Exact mirror coordinates matching cv2.flip(..., 1)
    pts_left_mirrored = np.column_stack(((w - 1) - x_l, y_l)).astype(np.float32)
    pts_right_mirrored = np.column_stack(((w - 1) - x_r, y_r)).astype(np.float32)

    # 3. Horizontal-Only Registration (tx only, ty=0, rot=0)
    M_R, aligned_left_on_right = compute_horizontal_registration(pts_left_mirrored, pts_right)
    M_L, aligned_right_on_left = compute_horizontal_registration(pts_right_mirrored, pts_left)

    flipped_full_mask = cv2.flip(full_mask, 1)
    flipped_full_gray = cv2.flip(full_gray, 1)

    # Warp mirrored references for Right hemisphere evaluation
    if M_R is not None:
        warped_mask_R = cv2.warpAffine(flipped_full_mask, M_R, (w, h), flags=cv2.INTER_NEAREST)
        warped_gray_R = cv2.warpAffine(flipped_full_gray, M_R, (w, h), flags=cv2.INTER_LINEAR)
    else:
        warped_mask_R = flipped_full_mask
        warped_gray_R = flipped_full_gray

    # Warp mirrored references for Left hemisphere evaluation
    if M_L is not None:
        warped_mask_L = cv2.warpAffine(flipped_full_mask, M_L, (w, h), flags=cv2.INTER_NEAREST)
        warped_gray_L = cv2.warpAffine(flipped_full_gray, M_L, (w, h), flags=cv2.INTER_LINEAR)
    else:
        warped_mask_L = flipped_full_mask
        warped_gray_L = flipped_full_gray

    # Reference gray aligned maps for intensity confirmation
    ref_gray_aligned = np.zeros((h, w), dtype=np.uint8)
    ref_gray_aligned[:, mid:] = warped_gray_R[:, mid:]
    ref_gray_aligned[:, :mid] = warped_gray_L[:, :mid]

    # 4. Multi-Scale Symmetrical Tolerance Subtraction
    # Multiple kernel sizes detect deficits across fine, medium, and large scales
    KERNEL_SCALES = [
        {"ksize": 7,  "weight": 0.40, "min_area": 300},   # Fine branch deficits
        {"ksize": 15, "weight": 0.70, "min_area": 500},   # Standard vessel deficits
        {"ksize": 27, "weight": 1.00, "min_area": 700},   # Major regional / trunk deficits
    ]

    # Canvas for composite multi-layer overlay
    overlay = np.zeros_like(output_img, dtype=np.float32)
    alpha_mask = np.zeros((h, w), dtype=np.float32)

    # Collect combined deficit mask across scales (used for dashboard & metrics)
    miss_mask_combined = np.zeros((h, w), dtype=np.uint8)
    cand_boxes = []
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 5. Deficit Detection & Intensity Confirmation across multiple kernel sizes
    for scale in KERNEL_SCALES:
        ksize = scale["ksize"]
        scale_weight = scale["weight"]
        min_area = scale["min_area"]

        tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_tol = cv2.dilate(full_mask, tol_kernel, iterations=1)

        miss_r = cv2.bitwise_and(warped_mask_R[:, mid:], cv2.bitwise_not(mask_tol[:, mid:]))
        miss_l = cv2.bitwise_and(warped_mask_L[:, :mid], cv2.bitwise_not(mask_tol[:, :mid]))

        scale_miss = np.zeros((h, w), dtype=np.uint8)
        scale_miss[:, mid:] = miss_r
        scale_miss[:, :mid] = miss_l

        cnts, _ = cv2.findContours(scale_miss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)

            # Intensity confirmation computed directly over deficit voxels (prevents background dilution)
            c_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(c_mask, [c - [bx, by]], -1, 255, -1)
            _m_pts = c_mask > 0

            roi_tgt = full_gray[by:by + bh, bx:bx + bw]
            roi_ref = ref_gray_aligned[by:by + bh, bx:bx + bw]

            # Regional contrast across the entire bounding ROI
            dn_rect = float(np.mean(roi_ref))
            ds_rect = float(np.mean(roi_tgt))
            rect_asym = max(0.0, (1.0 - (ds_rect / max(dn_rect, 1.0)))) * 100.0

            # Measure genuine regional vessel deficit within the ROI
            ref_mask_roi = warped_mask_R[by:by + bh, bx:bx + bw] if bx >= mid else warped_mask_L[by:by + bh, bx:bx + bw]
            tgt_mask_roi = full_mask[by:by + bh, bx:bx + bw]
            ref_v_px = int(np.count_nonzero(ref_mask_roi > 0))
            tgt_v_px = int(np.count_nonzero(tgt_mask_roi > 0))
            vessel_deficit = max(0.0, (ref_v_px - tgt_v_px) / max(1, ref_v_px)) * 100.0

            # Composite regional deficit: balanced combination of missing vessel volume & parenchymal contrast
            asym = 0.70 * vessel_deficit + 0.30 * rect_asym
            if asym < min_asym_thresh:
                continue

            # Tight multi-factor growth calculation:
            # 1. Deficit area growth (steep logistic ramp: 300px -> low, 1200px -> medium, 2500+px -> high)
            area_norm = (area - min_area) / 2500.0
            area_growth = 1.0 / (1.0 + np.exp(-2.5 * (area_norm - 0.6)))

            # 2. Regional contrast factor (rect_asym from 50% to 95%)
            rect_norm = np.clip((rect_asym - 50.0) / 45.0, 0.0, 1.0)

            # 3. Kernel scale weight
            scale_factor = scale_weight

            # Composite tight score
            raw_score = 0.50 * area_growth + 0.30 * rect_norm + 0.20 * scale_factor

            # Tight sigmoidal growth curve: steep transition around 0.50
            growth = 1.0 / (1.0 + np.exp(-8.0 * (raw_score - 0.50)))
            growth = float(np.clip(growth, 0.0, 1.0))

            # Mark in combined deficit mask
            cv2.drawContours(miss_mask_combined, [c], -1, 255, -1)

            # Opacity scales via tight growth curve: 0.35 (small/minor) to 0.80 (massive severe)
            base_alpha = (0.35 + 0.45 * growth) * (0.80 + 0.20 * scale_weight)

            # Consistent single-color red mask (BGR: 0, 0, 255)
            blob_color = (0.0, 0.0, 255.0)

            # Draw contour mask for this blob
            full_c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(full_c_mask, [c], -1, 255, -1)
            dil_miss = cv2.dilate(scale_miss, dil_k, iterations=1)
            blob_pixels = (full_c_mask > 0) & (dil_miss > 0)

            overlay[blob_pixels] = blob_color
            alpha_mask[blob_pixels] = np.maximum(alpha_mask[blob_pixels], base_alpha)

            # Collect candidates for high-deficit bounding box ("when it really high")
            # Triggers when growth is really high (>= 0.75) and area >= min_bbox_area
            if growth >= 0.65 and asym >= bbox_thresh and area >= min_area:
                cand_boxes.append((bx, by, bw, bh, asym, area))

    # Blend multi-kernel deficit overlay with original image (pure red mask only)
    alpha_3ch = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
    output_img = (overlay * alpha_3ch + output_img.astype(np.float32) * (1.0 - alpha_3ch)).clip(0, 255).astype(np.uint8)

    # 5c. Filtered high-deficit candidate regions
    merged_boxes = merge_boxes_compact(cand_boxes, iou_thresh=0.10, dist_thresh=12)
    filtered_boxes = [b for b in merged_boxes if b[4] >= bbox_thresh and b[5] >= min_bbox_area]

    # 6. Scientific Architecture Pipeline Output (6 Stages)
    # Panel 1: Auto-Crop ROI & Intracranial Head Isolation
    if meta is not None and "raw_clean" in meta and meta["raw_clean"] is not None:
        raw_bgr = meta["raw_clean"]
        raw_h, raw_w = raw_bgr.shape[:2]
        p1 = cv2.resize(raw_bgr, (w, h), interpolation=cv2.INTER_AREA)
        sx = w / float(max(1, raw_w))
        sy = h / float(max(1, raw_h))
        rx1, ry1, rx2, ry2 = meta.get("raw_crop_box", (0, 0, raw_w, raw_h))
        bx1, by1 = max(0, int(rx1 * sx)), max(0, int(ry1 * sy))
        bx2, by2 = min(w - 1, int(rx2 * sx)), min(h - 1, int(ry2 * sy))
        # Draw sleek crop bounding box
        cv2.rectangle(p1, (bx1, by1), (bx2, by2), (255, 230, 0), 2, cv2.LINE_AA)
        # Corner brackets
        blen = max(6, min(16, (bx2 - bx1) // 5))
        for cx_b, cy_b, dx, dy in [(bx1, by1, 1, 1), (bx2, by1, -1, 1), (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
            cv2.line(p1, (cx_b, cy_b), (cx_b + dx * blen, cy_b), (0, 240, 255), 2, cv2.LINE_AA)
            cv2.line(p1, (cx_b, cy_b), (cx_b, cy_b + dy * blen), (0, 240, 255), 2, cv2.LINE_AA)
        # Center crosshair
        ccx, ccy = int(meta.get("cx", raw_w // 2) * sx), int(meta.get("cy", raw_h // 2) * sy)
        cv2.drawMarker(p1, (ccx, ccy), (0, 240, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        sub_p1 = f"ROI: {bx2-bx1}x{by2-by1}px | Text Scrubbed"
    else:
        p1 = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
        pad_c = 14
        cv2.rectangle(p1, (pad_c, pad_c), (w - pad_c, h - pad_c), (255, 230, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(p1, (mid, h // 2), (0, 240, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        sub_p1 = "Intracranial Box | Text Scrubbed"
    draw_panel_card_header(p1, 1, "AUTO-CROP ROI", sub_p1)

    # Panel 2: Midline Axis & Auto-Tilt Correction
    tilt_deg = meta.get("angle", 0.0) if meta else 0.0
    effective_chest = meta.get("chest_y") if meta else None

    # Render on unrotated intracranial scan to visibly show anatomical tilt before rotation
    if meta is not None and "raw_clean" in meta and meta["raw_clean"] is not None:
        raw_bgr = meta["raw_clean"]
        raw_h, raw_w = raw_bgr.shape[:2]
        cx_val = int(meta.get("cx", raw_w // 2))
        cy_val = int(meta.get("cy", raw_h // 2))
        hw_val = int(meta.get("half_w", min(cx_val, raw_w - cx_val, cy_val, raw_h - cy_val)))

        top_u = max(0, cy_val - hw_val)
        bot_u = min(raw_h, cy_val + hw_val)
        left_u = max(0, cx_val - hw_val)
        right_u = min(raw_w, cx_val + hw_val)

        unrotated_crop = raw_bgr[top_u:bot_u, left_u:right_u]
        p2 = cv2.resize(unrotated_crop, (w, h), interpolation=cv2.INTER_AREA)

        scale_ux = w / float(max(1, right_u - left_u))
        scale_uy = h / float(max(1, bot_u - top_u))
        mid_x_ref = int((cx_val - left_u) * scale_ux)

        # 1. Strictly vertical 0.0° reference axis (dashed cyan line)
        for y_dash in range(36, h - 28, 10):
            cv2.line(p2, (mid_x_ref, y_dash), (mid_x_ref, min(h - 28, y_dash + 5)), (255, 230, 0), 1, cv2.LINE_AA)
        cv2.putText(p2, "0.0 DEG", (max(4, mid_x_ref - 52), 48), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 230, 0), 1, cv2.LINE_AA)

        # 2. Detected tilted midline axis (solid yellow line through anchor joints)
        pts_raw = meta.get("clicked_points", [])
        if len(pts_raw) >= 2:
            p2_joints = [((jx - left_u) * scale_ux, (jy - top_u) * scale_uy) for jx, jy in pts_raw]
            pts_np = np.array(p2_joints, dtype=np.float32)
            line_fit = cv2.fitLine(pts_np, cv2.DIST_L2, 0, 0.01, 0.01)
            vx_fit, vy_fit, x0_fit, y0_fit = float(line_fit[0][0]), float(line_fit[1][0]), float(line_fit[2][0]), float(line_fit[3][0])

            y_start = 36.0
            y_end = float(h - 28)
            x_start = int(x0_fit + (y_start - y0_fit) * (vx_fit / (vy_fit + 1e-7)))
            x_end = int(x0_fit + (y_end - y0_fit) * (vx_fit / (vy_fit + 1e-7)))
            cv2.line(p2, (x_start, int(y_start)), (x_end, int(y_end)), (0, 215, 255), 2, cv2.LINE_AA)

            # Glowing anchor joints on the tilted line
            for jx_p, jy_p in p2_joints:
                if 34 <= jy_p <= h - 26:
                    cv2.circle(p2, (int(jx_p), int(jy_p)), 5, (0, 215, 255), -1, cv2.LINE_AA)
                    cv2.circle(p2, (int(jx_p), int(jy_p)), 2, (18, 16, 20), -1, cv2.LINE_AA)

            # Rotation compensation arc / arrow near top
            arc_y = 66
            x_tilt_top = int(x0_fit + (arc_y - y0_fit) * (vx_fit / (vy_fit + 1e-7)))
            if abs(x_tilt_top - mid_x_ref) >= 2:
                cv2.arrowedLine(p2, (x_tilt_top, arc_y), (mid_x_ref, arc_y), (0, 140, 255), 2, tipLength=0.35)
                lbl_x = max(10, min(w - 110, (x_tilt_top + mid_x_ref) // 2 - 25))
                cv2.putText(p2, f"{tilt_deg:+.1f} DEG", (lbl_x, arc_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 160, 255), 1, cv2.LINE_AA)
        else:
            rad = math.radians(tilt_deg)
            x_top = int(mid_x_ref - (h // 2 - 36) * math.tan(rad))
            x_bot = int(mid_x_ref + (h - 28 - h // 2) * math.tan(rad))
            cv2.line(p2, (x_top, 36), (x_bot, h - 28), (0, 215, 255), 2, cv2.LINE_AA)

        # Chest cut indicator if active
        if effective_chest is not None and effective_chest < raw_h:
            y_chest_p2 = int((effective_chest - top_u) * scale_uy)
            if 36 < y_chest_p2 < h - 28:
                cv2.line(p2, (0, y_chest_p2), (w, y_chest_p2), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(p2, "CHEST CUT", (8, y_chest_p2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        p2 = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
        rad = math.radians(tilt_deg)
        x_top = int(mid - (h // 2 - 36) * math.tan(rad))
        x_bot = int(mid + (h - 28 - h // 2) * math.tan(rad))
        cv2.line(p2, (mid, 36), (mid, h - 28), (255, 230, 0), 1, cv2.LINE_AA)
        cv2.line(p2, (x_top, 36), (x_bot, h - 28), (0, 215, 255), 2, cv2.LINE_AA)
        for frac in [0.20, 0.40, 0.60, 0.80]:
            yj = int(h * frac)
            xj = int(mid + (yj - h // 2) * math.tan(rad))
            cv2.circle(p2, (xj, yj), 5, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(p2, (xj, yj), 2, (18, 16, 20), -1, cv2.LINE_AA)

    sub_p2 = f"Tilt: {tilt_deg:+.2f} deg -> Straightened" if abs(tilt_deg) > 0.05 else "FOV Symmetrical | Tilt: 0.0 deg"
    draw_panel_card_header(p2, 2, "MIDLINE & AUTO-TILT", sub_p2)

    # Panel 3: Vessel Segmentation & Pruned Skeleton Graph
    p3 = cv2.cvtColor((full_gray * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    p3[full_mask > 0] = (145, 170, 190)
    p3[clean_skel > 0] = (0, 240, 75)
    neighbor_map = convolve((clean_skel > 0).astype(np.uint8), NEIGHBOR_KERNEL, mode="constant", cval=0)
    for ey, ex in np.argwhere(neighbor_map == ENDPOINT_NEIGHBOR_SUM):
        cv2.circle(p3, (int(ex), int(ey)), 2, (0, 45, 255), -1)
    for by_node, bx_node in np.argwhere(neighbor_map >= 13):
        cv2.circle(p3, (int(bx_node), int(by_node)), 1, (255, 225, 0), -1)
    sub_p3 = "Top-Hat CLAHE | Pruned Graph"
    draw_panel_card_header(p3, 3, "VESSEL SKELETON", sub_p3)

    # Panel 4: Hemispheric Split & Contralateral Mirroring
    p4 = cv2.cvtColor((full_gray * 0.40).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Left hemisphere: Electric cyan vessel arbor
    p4[:, :mid][full_mask[:, :mid] > 0] = (245, 195, 45)
    p4[:, :mid][clean_skel[:, :mid] > 0] = (0, 255, 120)
    # Right hemisphere: Mirrored contralateral reference (Flipped across midline)
    flipped_clean_skel = cv2.flip(clean_skel, 1)
    p4[:, mid:][flipped_full_mask[:, mid:] > 0] = (220, 60, 215)
    p4[:, mid:][flipped_clean_skel[:, mid:] > 0] = (255, 140, 255)
    # Midline bisection axis
    cv2.line(p4, (mid, 34), (mid, h - 26), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(p4, "L-HEMI", (max(10, mid - 55), 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(p4, "MIRR-R", (mid + 8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 120, 245), 1, cv2.LINE_AA)
    sub_p4 = "Left (Cyan) vs Mirrored R (Pink)"
    draw_panel_card_header(p4, 4, "SPLIT & MIRROR", sub_p4)

    # Panel 5: Locked-Y 1D Horizontal Registration (Light Mode)
    p5 = np.full((h, w, 3), 248, dtype=np.uint8)
    p5[pts_right[:, 1].astype(int), pts_right[:, 0].astype(int)] = (160, 160, 165)
    p5[pts_left[:, 1].astype(int), pts_left[:, 0].astype(int)] = (160, 160, 165)
    if aligned_left_on_right is not None:
        val_r = (aligned_left_on_right[:, 0] >= 0) & (aligned_left_on_right[:, 0] < w) & \
                (aligned_left_on_right[:, 1] >= 0) & (aligned_left_on_right[:, 1] < h)
        p5[aligned_left_on_right[val_r, 1].astype(int), aligned_left_on_right[val_r, 0].astype(int)] = (20, 160, 40)
    if aligned_right_on_left is not None:
        val_l = (aligned_right_on_left[:, 0] >= 0) & (aligned_right_on_left[:, 0] < w) & \
                (aligned_right_on_left[:, 1] >= 0) & (aligned_right_on_left[:, 1] < h)
        p5[aligned_right_on_left[val_l, 1].astype(int), aligned_right_on_left[val_l, 0].astype(int)] = (220, 90, 0)
    reg_tx = float(M_R[0, 2]) if M_R is not None else 0.0
    sub_p5 = f"tx: {reg_tx:+.1f}px | ty: 0 (Locked)"
    draw_panel_card_header(p5, 5, "LOCKED-Y ALIGN", sub_p5)

    # Panel 6: Multi-Scale Deficit Mapping (Clean Arterial Heatmap)
    p6 = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
    alpha_3ch = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
    p6 = (overlay * alpha_3ch + p6.astype(np.float32) * (1.0 - alpha_3ch)).clip(0, 255).astype(np.uint8)

    # Hemispheric vessel masks (already symmetrically aligned along vertical axis mid = w // 2 by preprocess.py)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    left_mask[:, :mid] = full_mask[:, :mid]
    right_mask[:, mid:] = full_mask[:, mid:]

    left_px = int((left_mask > 0).sum())
    right_px = int((right_mask > 0).sum())
    tot_px = left_px + right_px
    asym_pct = abs(right_px - left_px) / float(max(1, tot_px)) * 100.0
    if asym_pct >= 5.0:
        aff_side = "RIGHT" if left_px > right_px else "LEFT"
    else:
        aff_side = "SYMMETRIC"
    sub_p6 = f"Vessel Asym: {asym_pct:.1f}% | Deficit: {aff_side}"
    draw_panel_card_header(p6, 6, "DEFICIT OVERLAY", sub_p6)

    profile = compute_asymmetry_profile(left_mask, right_mask, 0, h, n_bins=10)

    global LAST_ASYMMETRY_PROFILE, LAST_BOUNDING_BOXES
    LAST_ASYMMETRY_PROFILE = profile

    boxes_meta = [
        {
            "x": int(bx),
            "y": int(by),
            "w": int(bw),
            "h": int(bh),
            "asymmetry": round(float(asym), 1),
            "area": float(area),
        }
        for (bx, by, bw, bh, asym, area) in filtered_boxes
    ]
    LAST_BOUNDING_BOXES = boxes_meta

    all_panels = [p1, p2, p3, p4, p5, p6]
    combined_dash = np.hstack(all_panels)
    debug_dash = DashboardImage(
        combined_dash,
        profile=profile,
        bounding_boxes=boxes_meta,
        panels=all_panels,
        panel_titles=[
            "1. AUTO-CROP ROI",
            "2. MIDLINE & TILT",
            "3. VESSEL SKELETON",
            "4. SPLIT & MIRROR",
            "5. LOCKED-Y ALIGN",
            "6. DEFICIT OVERLAY"
        ],
        panel_subtitles=[
            sub_p1,
            sub_p2,
            sub_p3,
            sub_p4,
            sub_p5,
            sub_p6
        ],
        panel_descriptions=[
            "Brain ROI isolated & text scrubbed",
            "Midline axis & head tilt corrected",
            "Arterial tree & pruned skeleton",
            "Hemispheric split & mirror flipped",
            "Horizontal registration (locked Y)",
            "Multi-scale arterial deficit heatmap & mirror tolerance"
        ]
    )
    return output_img, debug_dash


# --- 4. Tilt-Robust Asymmetry Profile & Mirror Reference Gap Analysis ---

def midline_arc_length_position(midline_x, y_top=0, y_bottom=None):
    # Cumulative arc length along the midline curve, normalized to 0-1.
    # Used instead of raw row index so comparison zones stay anatomically
    # consistent regardless of neck tilt/length differences between scans.

    if y_bottom is None:
        y_bottom = len(midline_x)
    y_top = max(0, int(y_top))
    y_bottom = min(len(midline_x), int(y_bottom))
    if y_bottom <= y_top + 5:
        return np.linspace(0, 1, max(1, y_bottom - y_top))

    dx = np.gradient(midline_x[y_top:y_bottom])
    ds = np.sqrt(dx**2 + 1.0**2)
    arc = np.concatenate([[0], np.cumsum(ds)[:-1]])
    total = arc[-1] if arc[-1] > 0 else 1.0
    return arc / total


def compute_asymmetry_profile(left_mask, right_mask, *args, y_top=0, y_bottom=None, n_bins=10, **kwargs):
    # Per-depth-bin asymmetry index: AI = (R - L) / (R + L).
    # AI near 0 = symmetric. A large |AI| localized to one zone is a
    # candidate region of interest (screening signal).
    # Support legacy signature: compute_asymmetry_profile(left_mask, right_mask, midline_x, y_top, y_bottom, n_bins)
    if args:
        if len(args) == 4 and isinstance(args[1], (int, float)):
            y_top, y_bottom, n_bins = int(args[1]), int(args[2]), int(args[3])
        elif len(args) == 3 and isinstance(args[0], (int, float)):
            y_top, y_bottom, n_bins = int(args[0]), int(args[1]), int(args[2])

    h = left_mask.shape[0]
    if y_bottom is None:
        y_bottom = h
    y_top = max(0, int(y_top))
    y_bottom = min(h, int(y_bottom))
    if y_bottom <= y_top + 10:
        return []

    bin_edges = np.linspace(y_top, y_bottom, n_bins + 1, dtype=int)
    profile = []
    for b in range(n_bins):
        r_start, r_end = bin_edges[b], bin_edges[b + 1]
        l_count = int((left_mask[r_start:r_end, :] > 0).sum())
        r_count = int((right_mask[r_start:r_end, :] > 0).sum())
        total = l_count + r_count
        ai = (r_count - l_count) / total if total > 0 else 0.0
        profile.append(
            {
                "arc_range": (round(b / n_bins, 2), round((b + 1) / n_bins, 2)),
                "left_px": l_count,
                "right_px": r_count,
                "asymmetry_index": round(ai, 3),
            }
        )
    return profile


def print_asymmetry_profile(profile, flag_thresh=0.35):
    # Prints a clean tabular asymmetry report to console.
    if not profile:
        return
    print("\n  Anatomical Arc Range   Left (px)   Right (px)    AI      Screening Flag")
    print("  -----------------------------------------------------------------------")
    for row in profile:
        lo, hi = row["arc_range"]
        ai = row["asymmetry_index"]
        flag = "  <-- SIGNIFICANT DEFICIT" if abs(ai) >= flag_thresh else ""
        side = " (L>R)" if ai < -flag_thresh else (" (R>L)" if ai > flag_thresh else "")
        print(f"  Zone {lo:0.2f} - {hi:0.2f}          {row['left_px']:6d}      {row['right_px']:6d}   {ai:+.3f}  {flag}{side}")