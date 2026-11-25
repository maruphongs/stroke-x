import math
import os

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

# Globals for UI and mouse state
clicked_points = []
current_mouse = None
tilt_angle_offset = 0.0
warp_enabled = True
current_chest_y = None
auto_midline_curve = None
is_manual_mode = False


class ProcessedResult(tuple):
    # 5-tuple subclass for backward compatibility:
    # (left_tensor, right_tensor, left_high_res, right_high_res, crop_bgr)
    # with attached metadata (chest_y, midline_x, confidence, etc.).

    def __new__(cls, left_tensor, right_tensor, left_high, right_high, crop_bgr, meta=None):
        instance = super().__new__(cls, (left_tensor, right_tensor, left_high, right_high, crop_bgr))
        instance.left_tensor = left_tensor
        instance.right_tensor = right_tensor
        instance.left_high_res = left_high
        instance.right_high_res = right_high
        instance.crop_bgr = crop_bgr
        instance.meta = meta or {}
        return instance


def remove_patient_text(gray_img):
    """
    Finds the main central brain/skull structure and isolates it,
    completely removing peripheral UI text, icons, orientation markers,
    and artifacts from the margins.
    """
    _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return gray_img

    h, w = gray_img.shape
    img_center_x, img_center_y = w / 2.0, h / 2.0
    diag = np.hypot(img_center_x, img_center_y)

    def score_contour(c):
        area = cv2.contourArea(c)
        if area < 50:
            return -1e9
        M = cv2.moments(c)
        if M['m00'] == 0:
            return -1e9
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        norm_dist = np.hypot(cx - img_center_x, cy - img_center_y) / diag
        return np.sqrt(area) * (1.0 - 0.85 * norm_dist)

    main_c = max(contours, key=score_contour)
    hull = cv2.convexHull(main_c)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    clean_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    return clean_gray


# ============================================================
# 1. BRANCH ORIENTATION & CHEST GUILLOTINE
# ============================================================

def branch_angle_from_vertical(branch_pixels):
    """PCA-based orientation of a branch, in degrees from vertical."""
    pts = branch_pixels.astype(np.float64)
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    dy, dx = principal
    angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9))
    proj = pts @ principal
    length = proj.max() - proj.min()
    return angle, length


def row_width_chest_estimate(vessel_mask, search_start, h, width_ratio_thresh=1.9, sustain_rows=6, safety_margin=18):
    """
    Detects downward row width or lateral span expansion for aortic arch / subclavian vessels
    with a protective margin to eliminate trunk remnants below the neck.
    """
    widths = (vessel_mask > 0).sum(axis=1).astype(np.float64)
    spans = np.zeros(h, dtype=np.float64)
    for y in range(h):
        cols = np.where(vessel_mask[y, :] > 0)[0]
        if len(cols) >= 2:
            spans[y] = cols[-1] - cols[0]

    # Baseline neck measurement in cervical column
    neck_window = widths[search_start : min(h, search_start + int(h * 0.20))]
    valid_neck = neck_window[neck_window >= 6]
    baseline = np.median(valid_neck) if len(valid_neck) else 0
    if baseline <= 0:
        return None

    span_window = spans[search_start : min(h, search_start + int(h * 0.20))]
    valid_spans = span_window[span_window >= 15]
    baseline_span = np.median(valid_spans) if len(valid_spans) else 0

    width_threshold = baseline * width_ratio_thresh
    span_threshold = max(baseline_span * 2.0, 320.0) if baseline_span > 0 else 1e9

    count = 0
    for y in range(search_start, h):
        if widths[y] > width_threshold or spans[y] > span_threshold:
            count += 1
            if count >= sustain_rows:
                detected_y = y - sustain_rows + 1
                return max(search_start + 10, detected_y)
        else:
            count = 0

    # Vascular column termination (e.g. scans where neck vessels end without full chest)
    if len(valid_neck) >= 5:
        for y in range(search_start + 20, h):
            if y + 10 <= h and np.all(widths[y : y + 10] <= 4):
                return y + 2

    return None


def detect_chest_guillotine(vessel_mask, isolated_branches, h, w, fov_type=None):
    """
    Combines PCA branch orientation analysis, lateral span/width expansion,
    and vascular column termination to cleanly sever the chest boundary (chest_y).
    """
    if fov_type == "HEAD_ONLY":
        return h, None, None

    search_start = int(h * 0.54)
    branch_chest_y = h
    ANGLE_THRESH_DEG = 72
    MIN_LENGTH = 30

    num_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(
        isolated_branches, connectivity=8
    )
    for i in range(1, num_b):
        if stats_b[i, cv2.CC_STAT_AREA] < 8:
            continue
        branch_pixels = np.argwhere(labels_b == i)
        min_y = np.min(branch_pixels[:, 0])
        if min_y <= search_start or len(branch_pixels) < 3:
            continue
        angle, length = branch_angle_from_vertical(branch_pixels)
        if (angle > ANGLE_THRESH_DEG and length > MIN_LENGTH) and (min_y < branch_chest_y):
            branch_chest_y = min_y - 15

    width_chest_y = row_width_chest_estimate(vessel_mask, search_start, h)

    if width_chest_y is not None and branch_chest_y < h:
        if abs(width_chest_y - branch_chest_y) <= 50:
            chest_y = min(width_chest_y, branch_chest_y)
        else:
            chest_y = width_chest_y
    elif width_chest_y is not None:
        chest_y = width_chest_y
    elif branch_chest_y < h:
        chest_y = branch_chest_y
    else:
        chest_y = h

    return chest_y, width_chest_y, branch_chest_y


# ============================================================
# 2. ANCHOR-TREND WEIGHTED POLYNOMIAL MIDLINE ESTIMATOR
# ============================================================

def estimate_midline(
    mask,
    y_top,
    y_bottom,
    min_gap=3,
    degree=2,
    outlier_rounds=2,
    anchor_frac=0.4,
    sigma=30.0,
    is_head_only=False,
):
    """
    Single continuous curve fit for the left/right midline ("spine proxy").
    Uses robust linear anchor trend and soft Gaussian distance weighting.
    - If is_head_only: fits linear trend across cranial height with wider tolerance (sigma=36).
    - If FULL_ARCH: anchors on lower neck trunk with responsive sigma=30 to allow strong cranial tilt.
    """
    h, w = mask.shape
    y_top = max(0, int(y_top))
    y_bottom = min(h, int(y_bottom))
    if y_bottom <= y_top:
        y_top, y_bottom = 0, h

    def row_candidates(y):
        row = mask[y, :] > 0
        if not row.any():
            return None
        cols = np.where(row)[0]
        splits = np.where(np.diff(cols) > 1)[0]
        segments = np.split(cols, splits + 1)
        if len(segments) < 2:
            return None
        segments = sorted(segments, key=len, reverse=True)[:2]
        segments = sorted(segments, key=lambda s: s[0])
        left_seg, right_seg = segments
        gap = right_seg[0] - left_seg[-1]
        if gap < min_gap:
            return None
        mid = (left_seg[-1] + right_seg[0]) / 2.0
        weight = min(len(left_seg), len(right_seg))
        return mid, weight

    ys, xs, ws = [], [], []
    for y in range(y_top, y_bottom):
        c = row_candidates(y)
        if c is not None:
            ys.append(y)
            xs.append(c[0])
            ws.append(c[1])

    if len(xs) < degree + 1:
        _ys_all, xs_all = np.where(mask > 0)
        cx = xs_all.mean() if len(xs_all) else w / 2.0
        return np.full(h, cx, dtype=np.float64)

    ys = np.array(ys, dtype=np.float64)
    xs = np.array(xs, dtype=np.float64)
    ws = np.array(ws, dtype=np.float64)

    if not is_head_only:
        # Reference trend from the neck trunk only
        anchor_y_cut = y_top + anchor_frac * (y_bottom - y_top)
        anchor = ys >= anchor_y_cut

        if anchor.sum() >= 4:
            ay, ax, aw = ys[anchor], xs[anchor], ws[anchor]
            for _ in range(outlier_rounds):
                coeffs = np.polyfit(ay, ax, 1, w=np.sqrt(aw))
                resid = np.abs(ax - np.polyval(coeffs, ay))
                thresh = max(4.0, 2.5 * np.median(resid))
                keep = resid <= thresh
                if keep.sum() < 4 or keep.all():
                    break
                ay, ax, aw = ay[keep], ax[keep], aw[keep]
            trend_coeffs = np.polyfit(ay, ax, 1, w=np.sqrt(aw))
        else:
            trend_coeffs = np.polyfit(ys, xs, 1, w=np.sqrt(ws))

        expected_x = np.polyval(trend_coeffs, ys)
        dist = xs - expected_x
        soft_factor = np.exp(-(dist**2) / (2.0 * sigma**2))
        combined_w = ws * soft_factor
    else:
        # For head-only scans: linear trend across entire cranial height
        ay, ax, aw = ys.copy(), xs.copy(), ws.copy()
        for _ in range(outlier_rounds):
            coeffs = np.polyfit(ay, ax, 1, w=np.sqrt(aw))
            resid = np.abs(ax - np.polyval(coeffs, ay))
            thresh = max(4.0, 2.5 * np.median(resid))
            keep = resid <= thresh
            if keep.sum() < 4 or keep.all():
                break
            ay, ax, aw = ay[keep], ax[keep], aw[keep]
        trend_coeffs = np.polyfit(ay, ax, 1, w=np.sqrt(aw))
        expected_x = np.polyval(trend_coeffs, ys)
        dist = xs - expected_x
        soft_factor = np.exp(-(dist**2) / (2.0 * (sigma * 1.4)**2))
        combined_w = ws * soft_factor

    # Polynomial fit with outlier rejection
    fys, fxs, fws = ys.copy(), xs.copy(), combined_w.copy()
    for _ in range(outlier_rounds):
        coeffs = np.polyfit(fys, fxs, degree, w=np.sqrt(fws + 1e-6))
        resid = np.abs(fxs - np.polyval(coeffs, fys))
        thresh = max(4.0, 2.5 * np.median(resid))
        keep = resid <= thresh
        if keep.sum() < degree + 1 or keep.all():
            break
        fys, fxs, fws = fys[keep], fxs[keep], fws[keep]

    coeffs = np.polyfit(fys, fxs, degree, w=np.sqrt(fws + 1e-6))
    y_idx = np.arange(h)
    return np.clip(np.polyval(coeffs, y_idx), 0, w - 1)


# ============================================================
# 3. AUTOMATED MIDLINE & CHEST DETECTOR
# ============================================================

def detect_head_bottom_vessel_center(clean_gray):
    """
    For HEAD_ONLY scans: locates the horizontal center (cx) based on the
    bottom area of the vessels (skull base / basilar / vertebral trunks).
    Does NOT tilt or rotate the image; strictly returns 0.0 tilt and shifted center.
    """
    h, w = clean_gray.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(clean_gray)
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, k_large)
    _, binary = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    vessel_mask = np.zeros_like(binary)
    if num > 1:
        vessel_mask[labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] = 255

    ys, xs = np.where(vessel_mask > 0)
    if len(ys) < 30:
        return round(w / 2.0), round(h / 2.0)

    y_min, y_max = int(ys.min()), int(ys.max())
    bot_y0 = y_min + int((y_max - y_min) * 0.70)
    bot_y1 = y_max

    row_centers = []
    for y in range(bot_y0, bot_y1 + 1):
        cols = np.where(vessel_mask[y, :] > 0)[0]
        if len(cols) >= 3:
            row_centers.append((cols[0] + cols[-1]) / 2.0)

    if row_centers:
        cx = float(np.median(row_centers))
    else:
        cx = float(np.median(xs))

    # Gentle clamp within 15% of image center to avoid edge noise
    max_shift = 0.15 * w
    cx = float(np.clip(cx, w / 2.0 - max_shift, w / 2.0 + max_shift))
    cy = float(np.clip((y_min + y_max) / 2.0, 0.30 * h, 0.70 * h))

    return round(cx), round(cy)


def auto_detect_midline_and_chest(clean_gray, fov_type=None, num_joints=8):
    """
    Automated pipeline orchestrator:
    - If HEAD_ONLY: runs skull/brain vessel tree extraction, active midline estimation,
      and tilt detection (adaptive cranial trend), with chest_y = h. Falls back to bottom
      vessel center only if vessel mask is too sparse.
    - If FULL_ARCH: runs chest guillotine and anchor-trend polynomial midline estimator,
      with cranial tilt blending for natural head-on-neck posture.
    """
    h, w = clean_gray.shape
    is_head = (fov_type == "HEAD_ONLY")

    # Margin cleaning: slightly tighter for head scans so cortical vessels aren't clipped
    margin_x = int(w * 0.10 if is_head else w * 0.15)
    margin_y = int(h * 0.10 if is_head else h * 0.15)
    margin_cleaned = clean_gray.copy()
    cv2.rectangle(margin_cleaned, (0, 0), (w - 1, h - 1), 0, 5)
    margin_cleaned[:margin_y, :margin_x] = 0
    margin_cleaned[:margin_y, -margin_x:] = 0
    margin_cleaned[-margin_y:, :margin_x] = 0
    margin_cleaned[-margin_y:, -margin_x:] = 0

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(margin_cleaned)
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, k_large)
    _, binary = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    vessel_mask = np.zeros_like(binary)
    if num > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        vessel_mask[labels == largest_label] = 255

    vessel_px = int((vessel_mask > 0).sum())
    min_required_px = 250 if is_head else 400
    if vessel_px < min_required_px:
        if is_head:
            # Fallback to bottom vessel center for sparse head scan
            cx, cy = detect_head_bottom_vessel_center(clean_gray)
            joints = [(cx, int(h * frac)) for frac in np.linspace(0.15, 0.85, num_joints)]
            midline_x = np.full(h, cx, dtype=np.float64)
            meta = {
                "chest_y": h, "width_chest_y": None, "branch_chest_y": None,
                "angle": 0.0, "cx": cx, "cy": cy, "y_top": 0, "y_bottom": h,
                "fov_type": "HEAD_ONLY", "confidence": 0.60, "shift_x": cx - (w / 2.0),
            }
            return joints, h, midline_x, 0.60, meta
        return None, h, None, 0.0, {"error": "Insufficient vessel coverage"}

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smoothed = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, close_k)
    skel = skeletonize(smoothed > 0).astype(np.uint8)

    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_map = convolve(skel, kernel, mode="constant", cval=0)
    isolated_branches = skel.copy()
    isolated_branches[neighbor_map >= 13] = 0

    # Chest guillotine detection (only for FULL_ARCH)
    if is_head:
        chest_y = h
        width_chest_y = None
        branch_chest_y = None
    else:
        chest_y, width_chest_y, branch_chest_y = detect_chest_guillotine(
            vessel_mask, isolated_branches, h, w, fov_type=fov_type
        )

    clean_vessel_mask = vessel_mask.copy()
    if chest_y < h:
        clean_vessel_mask[chest_y:, :] = 0

    ys, _xs = np.where(clean_vessel_mask > 0)
    if len(ys) < 80:
        if is_head:
            cx, cy = detect_head_bottom_vessel_center(clean_gray)
            joints = [(cx, int(h * frac)) for frac in np.linspace(0.15, 0.85, num_joints)]
            midline_x = np.full(h, cx, dtype=np.float64)
            meta = {
                "chest_y": h, "width_chest_y": None, "branch_chest_y": None,
                "angle": 0.0, "cx": cx, "cy": cy, "y_top": 0, "y_bottom": h,
                "fov_type": "HEAD_ONLY", "confidence": 0.50, "shift_x": cx - (w / 2.0),
            }
            return joints, h, midline_x, 0.50, meta
        return None, chest_y, None, 0.0, {"error": "Vessel mask too sparse after guillotine cut"}

    y_top = max(0, int(np.percentile(ys, 1)))
    y_bottom = min(h - 1, int(chest_y if chest_y < h else np.percentile(ys, 99)))

    if y_bottom - y_top < 50:
        return None, chest_y, None, 0.0, {"error": "Vertical vessel span too small"}

    midline_x = estimate_midline(
        clean_vessel_mask,
        y_top,
        y_bottom,
        degree=2,
        anchor_frac=0.30 if is_head else 0.40,
        sigma=36.0 if is_head else 30.0,
        is_head_only=is_head,
    )

    # Sample num_joints evenly along midline_x between anatomical top and base
    sample_ys = np.linspace(y_top + 10, y_bottom - 10, num_joints).astype(int)
    joints = [(round(midline_x[y]), int(y)) for y in sample_ys]

    in_bounds = all(20 <= pt[0] < w - 20 and 0 <= pt[1] < h for pt in joints)
    angle, cx, cy = compute_midline_tilt(joints)

    # In FULL_ARCH, if the upper cranial portion has stronger tilt than the neck trunk,
    # blend towards the cranial angle to ensure the head vessels are properly straightened
    if not is_head and len(joints) >= 6:
        upper_joints = joints[: len(joints) // 2 + 1]
        upper_angle, _, _ = compute_midline_tilt(upper_joints)
        if abs(upper_angle) > abs(angle) + 0.8:
            angle = 0.60 * upper_angle + 0.40 * angle

    tilt_ok = abs(angle) <= 30.0
    span_ratio = (y_bottom - y_top) / float(h)
    span_ok = span_ratio >= 0.25

    confidence = 0.50
    if in_bounds:
        confidence += 0.20
    if tilt_ok:
        confidence += 0.20
    if span_ok:
        confidence += 0.10

    if not in_bounds or not tilt_ok:
        confidence = min(confidence, 0.35)

    meta = {
        "chest_y": chest_y,
        "width_chest_y": width_chest_y,
        "branch_chest_y": branch_chest_y,
        "angle": angle,
        "cx": cx,
        "cy": cy,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "vessel_px": vessel_px,
        "span_ratio": span_ratio,
        "confidence": confidence,
        "fov_type": fov_type or ("HEAD_ONLY" if is_head else "FULL_ARCH"),
    }

    return joints, chest_y, midline_x, confidence, meta


# ============================================================
# 4. EXISTING GEOMETRIC TRANSFORMS & WARPING
# ============================================================

def compute_midline_tilt(points, angle_offset=0.0):
    """
    Computes the tilt angle (in degrees from vertical) and rotation center (cx, cy)
    from a list of joints/points using robust linear fitting.
    """
    if len(points) < 2:
        return 0.0, 0, 0

    pts_arr = np.array(points, dtype=np.float32)
    if len(points) == 2:
        dx = float(pts_arr[1, 0] - pts_arr[0, 0])
        dy = float(pts_arr[1, 1] - pts_arr[0, 1])
        if dy < 0:
            dx, dy = -dx, -dy
        angle = math.degrees(math.atan2(dx, max(abs(dy), 1e-6)))
        cx = round((pts_arr[0, 0] + pts_arr[1, 0]) / 2.0)
        cy = round((pts_arr[0, 1] + pts_arr[1, 1]) / 2.0)
    else:
        line_params = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(line_params[0][0])
        vy = float(line_params[1][0])
        if vy < 0:
            vx, vy = -vx, -vy
        angle = math.degrees(math.atan2(vx, max(abs(vy), 1e-6)))
        cx = round(np.mean(pts_arr[:, 0]))
        cy = round(np.mean(pts_arr[:, 1]))

    total_angle = angle + angle_offset
    return total_angle, cx, cy


def warp_straighten_midline(img, rot_joints, cx):
    """
    Non-rigid distortion / warping that straightens a curved midline
    into a vertical line at x = cx, preserving anatomical structure.
    """
    h, w = img.shape[:2]
    pts = np.array(rot_joints, dtype=np.float32)
    pts = pts[np.argsort(pts[:, 1])]  # Sort by y

    if len(pts) < 2:
        return img.copy(), np.full(h, cx, dtype=np.float32)

    slope_top = (pts[1, 0] - pts[0, 0]) / max(pts[1, 1] - pts[0, 1], 1e-3)
    slope_bot = (pts[-1, 0] - pts[-2, 0]) / max(pts[-1, 1] - pts[-2, 1], 1e-3)

    x_mid = np.empty(h, dtype=np.float32)
    for y in range(h):
        if y < pts[0, 1]:
            x_mid[y] = pts[0, 0] + slope_top * (y - pts[0, 1])
        elif y > pts[-1, 1]:
            x_mid[y] = pts[-1, 0] + slope_bot * (y - pts[-1, 1])
        else:
            x_mid[y] = np.interp(y, pts[:, 1], pts[:, 0])

    ksize = min(21, (h // 16) * 2 + 1)
    if ksize >= 3:
        x_mid = cv2.GaussianBlur(x_mid.reshape(-1, 1), (1, ksize), 3.0).flatten()

    shift = (x_mid - cx).astype(np.float32)
    all_y = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), all_y)
    map_x = (grid_x + shift[:, None]).astype(np.float32)
    map_y = grid_y.astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, x_mid


# ============================================================
# 5. INTERACTIVE UI RENDERING
# ============================================================

def redraw_ui(
    img_clean,
    points,
    mouse_pos,
    angle_offset=0.0,
    warp_on=True,
    chest_y=None,
    midline_curve=None,
    manual_mode=False,
):
    """Renders placed joints, chest cut, midline curve guide, PIP preview, and HUD."""
    canvas = img_clean.copy()
    h, w = canvas.shape[:2]

    # Dim region below chest cut if chest guillotine is active
    if chest_y is not None and chest_y < h:
        chest_y_int = int(np.clip(chest_y, 0, h - 1))
        below_cut = canvas[chest_y_int:h, :]
        canvas[chest_y_int:h, :] = (below_cut * 0.45).astype(np.uint8)

        # Draw chest guillotine cut line
        cv2.line(canvas, (0, chest_y_int), (w, chest_y_int), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"[c] Chest Guillotine: y={chest_y_int}",
            (14, chest_y_int - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 100, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw continuous midline polynomial curve if available
    if midline_curve is not None and len(midline_curve) == h:
        step = max(2, h // 250)
        curve_pts = []
        for y in range(0, h, step):
            x = round(midline_curve[y])
            if 0 <= x < w:
                curve_pts.append((x, y))
        if len(curve_pts) > 1:
            for i in range(len(curve_pts) - 1):
                cv2.line(canvas, curve_pts[i], curve_pts[i + 1], (180, 220, 0), 1, cv2.LINE_AA)

    # Draw connecting segments between joints
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i + 1], (0, 230, 120), 2, cv2.LINE_AA)

    # Draw joints
    for idx, pt in enumerate(points):
        cv2.circle(canvas, pt, 7, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"J{idx + 1}",
            (pt[0] + 9, pt[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    # Dynamic mouse line
    if mouse_pos is not None and len(points) > 0 and manual_mode:
        cv2.line(canvas, points[-1], mouse_pos, (180, 255, 50), 1, cv2.LINE_AA)
        cv2.circle(canvas, mouse_pos, 4, (180, 255, 50), -1, cv2.LINE_AA)

    # Tilt & PIP preview
    angle, cx, cy = 0.0, 0, 0
    if len(points) >= 2:
        angle, cx, cy = compute_midline_tilt(points, angle_offset)
        rad = math.radians(angle)
        length = max(h, w) * 1.5
        p1 = (int(cx - length * math.sin(rad)), int(cy - length * math.cos(rad)))
        p2 = (int(cx + length * math.sin(rad)), int(cy + length * math.cos(rad)))

        cv2.line(canvas, p1, p2, (255, 160, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

        pip_size = 170
        pip_pad = 12
        if h > pip_size + 80 and w > pip_size + 80:
            thumb = cv2.resize(img_clean, (pip_size, pip_size), interpolation=cv2.INTER_AREA)
            scale_x = pip_size / float(w)
            scale_y = pip_size / float(h)
            thumb_cx = round(cx * scale_x)
            thumb_cy = round(cy * scale_y)

            rot_pip = cv2.getRotationMatrix2D((thumb_cx, thumb_cy), -angle, 1.0)
            pip_straight = cv2.warpAffine(thumb, rot_pip, (pip_size, pip_size), borderValue=(20, 20, 20))

            pip_label = f"Tilt ({angle:+.1f} deg)"
            if warp_on and len(points) >= 3:
                scaled_pts = [(p[0] * scale_x, p[1] * scale_y) for p in points]
                pts_arr = np.array(scaled_pts, dtype=np.float32).reshape(-1, 1, 2)
                pip_rot_pts = cv2.transform(pts_arr, rot_pip).reshape(-1, 2)
                pip_straight, _ = warp_straighten_midline(pip_straight, pip_rot_pts, thumb_cx)
                pip_label = f"Straightened ({angle:+.1f} deg)"

            cv2.line(pip_straight, (thumb_cx, 0), (thumb_cx, pip_size), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(pip_straight, (0, 0), (pip_size - 1, pip_size - 1), (0, 200, 255), 1)
            cv2.putText(
                pip_straight,
                pip_label,
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            canvas[
                h - pip_size - pip_pad - 36 : h - pip_pad - 36,
                w - pip_size - pip_pad : w - pip_pad,
            ] = pip_straight

    # Top HUD Banner (Light Mode)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.line(canvas, (0, 42), (w, 42), (210, 215, 225), 1)

    mode_badge = "[MANUAL EDIT]" if manual_mode else "[AUTO-DETECTED]"
    chest_badge = f" | Chest Cut: y={chest_y}" if (chest_y is not None and chest_y < h) else " | Chest: Head Only"
    nudge_badge = f" [nudge: {angle_offset:+.1f} deg]" if abs(angle_offset) > 1e-3 else ""
    warp_badge = f" | Warp: {'ON' if warp_on else 'OFF'}" if len(points) >= 3 else ""
    status_text = f"{mode_badge} Joints: {len(points)} | Tilt: {angle:+.1f} deg{nudge_badge}{warp_badge}{chest_badge}"

    cv2.putText(canvas, status_text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (15, 23, 42), 1, cv2.LINE_AA)

    # Bottom HUD Banner (Light Mode)
    overlay_bot = canvas.copy()
    cv2.rectangle(overlay_bot, (0, h - 32), (w, h), (255, 255, 255), -1)
    cv2.addWeighted(overlay_bot, 0.90, canvas, 0.10, 0, canvas)
    cv2.line(canvas, (0, h - 32), (w, h - 32), (210, 215, 225), 1)
    controls_text = (
        "[Click] Add Joint  [u] Undo  [c/v] Cut Row  [a/d] Tilt  [w] Warp  [m] Mode  [r] Reset  [Enter] Confirm"
    )
    cv2.putText(canvas, controls_text, (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (71, 85, 105), 1, cv2.LINE_AA)

    return canvas


def mouse_callback(event, x, y, flags, param):
    """Handles joint addition, removal, and hover tracking."""
    global clicked_points, current_mouse, tilt_angle_offset, warp_enabled
    global current_chest_y, auto_midline_curve, is_manual_mode
    img_clean, window_name = param

    if event == cv2.EVENT_LBUTTONDOWN:
        is_manual_mode = True
        clicked_points.append((x, y))
        rendered = redraw_ui(
            img_clean,
            clicked_points,
            current_mouse,
            tilt_angle_offset,
            warp_enabled,
            current_chest_y,
            auto_midline_curve,
            is_manual_mode,
        )
        cv2.imshow(window_name, rendered)

    elif event == cv2.EVENT_RBUTTONDOWN:
        is_manual_mode = True
        if clicked_points:
            clicked_points.pop()
        rendered = redraw_ui(
            img_clean,
            clicked_points,
            current_mouse,
            tilt_angle_offset,
            warp_enabled,
            current_chest_y,
            auto_midline_curve,
            is_manual_mode,
        )
        cv2.imshow(window_name, rendered)

    elif event == cv2.EVENT_MOUSEMOVE:
        current_mouse = (x, y)
        rendered = redraw_ui(
            img_clean,
            clicked_points,
            current_mouse,
            tilt_angle_offset,
            warp_enabled,
            current_chest_y,
            auto_midline_curve,
            is_manual_mode,
        )
        cv2.imshow(window_name, rendered)


# ============================================================
# 6. MAIN PREPROCESSING WORKFLOW (AUTO + MANUAL FALLBACK)
# ============================================================

def process_image(
    image_path_or_bgr,
    target_size=(224, 224),
    mode="auto",
    interactive_fallback=True,
    fov_type=None,
    preview=False,
):
    """
    Standard preprocessing pipeline:
    - Automatically obtains midline joints:
      * For HEAD_ONLY: uses centered vertical line directly without curve search.
      * For FULL_ARCH: uses PCA guillotine and anchor-trend polynomial midline estimator.
    - If manual mode or fallback: interactive GUI pre-populated with joints.
    - Applies existing method:
      * Head tilt compensation (warpAffine rotation)
      * Non-rigid midline straightening (warp_straighten_midline)
      * Chest guillotine cut (zeroes out pixels below cut)
      * Strict symmetrical square crop centered at (cx, cy)
      * Split down center and mirror right hemisphere
    """
    global clicked_points, current_mouse, tilt_angle_offset, warp_enabled
    global current_chest_y, auto_midline_curve, is_manual_mode

    clicked_points = []
    current_mouse = None
    tilt_angle_offset = 0.0
    warp_enabled = True
    current_chest_y = None
    auto_midline_curve = None
    is_manual_mode = (mode == "manual")

    # Load image
    if isinstance(image_path_or_bgr, str):
        img = cv2.imread(image_path_or_bgr)
        image_name = os.path.basename(image_path_or_bgr)
        if img is None:
            print(f"Error: Could not load image from '{image_path_or_bgr}'")
            return None
    else:
        img = image_path_or_bgr
        image_name = "in_memory_image"

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clean_gray = remove_patient_text(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(clean_gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    img_clean = enhanced_bgr.copy()

    # Auto-classify FOV if not specified
    if fov_type is None and os.path.exists("fov_model.pt"):
        try:
            from fov_ai import classify_fov
            fov_type, _ = classify_fov(img, model_weights="fov_model.pt")
        except Exception:
            fov_type = None

    # Step 1: Obtain midline joints and chest cut
    auto_joints, chest_y, midline_x, confidence, meta = auto_detect_midline_and_chest(
        clean_gray, fov_type=fov_type, num_joints=6
    )

    current_chest_y = chest_y
    auto_midline_curve = midline_x

    if auto_joints is not None:
        clicked_points = list(auto_joints)
        chest_info = f"chest_y={chest_y}" if (chest_y < h) else "no chest cut"
        print(
            f"[Midline Detection] {fov_type or 'AUTO'} | Confidence: {confidence*100:.1f}% | "
            f"Tilt: {meta['angle']:+.1f} deg | {chest_info} | {len(auto_joints)} joints"
        )
    else:
        err_msg = meta.get("error", "Unknown") if meta else "Failed"
        print(f"[Midline Detection] Low Confidence / Failed ({err_msg})")

    # Determine whether interactive GUI is required
    should_launch_ui = False
    if mode == "manual":
        should_launch_ui = True
        is_manual_mode = True
    elif mode == "auto":
        if confidence < 0.70 or len(clicked_points) < 2:
            if interactive_fallback:
                print(
                    f"[!] Low confidence ({confidence*100:.1f}%). "
                    "Launching interactive manual fallback."
                )
                should_launch_ui = True
                is_manual_mode = True
            else:
                print("[!] Auto-detection failed and interactive fallback is disabled.")
                return None
        elif preview:
            should_launch_ui = True
            is_manual_mode = False

    # Step 2: Interactive UI if requested or needed as fallback
    if should_launch_ui:
        window_name = f"Midline Alignment - {image_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback, param=(img_clean, window_name))

        display = redraw_ui(
            img_clean,
            clicked_points,
            current_mouse,
            tilt_angle_offset,
            warp_enabled,
            current_chest_y,
            auto_midline_curve,
            is_manual_mode,
        )
        cv2.imshow(window_name, display)

        print("\n--- Interactive Alignment Controls ---")
        print("  • Left-Click   : Add / replace joint")
        print("  • Right-Click/u: Undo last joint")
        print("  • c / v        : Nudge chest cut up (-15px) / down (+15px)")
        print("  • a / d        : Nudge tilt angle +/- 0.5 deg")
        print("  • w            : Toggle non-rigid distortion / midline straightening (ON/OFF)")
        print("  • m            : Switch between Auto and Manual edit mode")
        print("  • r            : Reset joints")
        print("  • Enter / Space: Confirm and proceed")
        print("  • q / Esc      : Abort / Quit")

        while True:
            raw_key = cv2.waitKeyEx(30)
            if raw_key == -1:
                continue

            key = raw_key & 0xFF

            if key == ord("q") or raw_key == 27:
                cv2.destroyAllWindows()
                return "QUIT"

            elif key in (13, ord(" "), 10):
                if len(clicked_points) >= 2:
                    break
                else:
                    print("[!] Please place at least 2 midline joints before confirming.")

            elif key == ord("m"):
                is_manual_mode = not is_manual_mode
                display = redraw_ui(
                    img_clean,
                    clicked_points,
                    current_mouse,
                    tilt_angle_offset,
                    warp_enabled,
                    current_chest_y,
                    auto_midline_curve,
                    is_manual_mode,
                )
                cv2.imshow(window_name, display)

            elif key == ord("c"):
                if current_chest_y is None or current_chest_y >= h:
                    current_chest_y = int(h * 0.70)
                else:
                    current_chest_y = max(int(h * 0.30), current_chest_y - 15)
                display = redraw_ui(
                    img_clean,
                    clicked_points,
                    current_mouse,
                    tilt_angle_offset,
                    warp_enabled,
                    current_chest_y,
                    auto_midline_curve,
                    is_manual_mode,
                )
                cv2.imshow(window_name, display)

            elif key in (ord("v"), ord("C")):
                if current_chest_y is not None:
                    current_chest_y = min(h, current_chest_y + 15)
                display = redraw_ui(
                    img_clean,
                    clicked_points,
                    current_mouse,
                    tilt_angle_offset,
                    warp_enabled,
                    current_chest_y,
                    auto_midline_curve,
                    is_manual_mode,
                )
                cv2.imshow(window_name, display)

            elif key == ord("w"):
                warp_enabled = not warp_enabled
                display = redraw_ui(
                    img_clean,
                    clicked_points,
                    current_mouse,
                    tilt_angle_offset,
                    warp_enabled,
                    current_chest_y,
                    auto_midline_curve,
                    is_manual_mode,
                )
                cv2.imshow(window_name, display)

            elif key == ord("r"):
                clicked_points.clear()
                tilt_angle_offset = 0.0
                is_manual_mode = True
                display = redraw_ui(
                    img_clean,
                    clicked_points,
                    current_mouse,
                    tilt_angle_offset,
                    warp_enabled,
                    current_chest_y,
                    auto_midline_curve,
                    is_manual_mode,
                )
                cv2.imshow(window_name, display)

            elif key == ord("u") or key == 8 or raw_key == 8:
                if clicked_points:
                    clicked_points.pop()
                is_manual_mode = True
                display = redraw_ui(
                    img_clean,
                    clicked_points,
                    current_mouse,
                    tilt_angle_offset,
                    warp_enabled,
                    current_chest_y,
                    auto_midline_curve,
                    is_manual_mode,
                )
                cv2.imshow(window_name, display)

            elif key == ord("a") or key == ord("[") or raw_key in (2424832, 81, 65361):
                if len(clicked_points) >= 2:
                    tilt_angle_offset -= 0.5
                    display = redraw_ui(
                        img_clean,
                        clicked_points,
                        current_mouse,
                        tilt_angle_offset,
                        warp_enabled,
                        current_chest_y,
                        auto_midline_curve,
                        is_manual_mode,
                    )
                    cv2.imshow(window_name, display)

            elif key == ord("d") or key == ord("]") or raw_key in (2555904, 83, 65363):
                if len(clicked_points) >= 2:
                    tilt_angle_offset += 0.5
                    display = redraw_ui(
                        img_clean,
                        clicked_points,
                        current_mouse,
                        tilt_angle_offset,
                        warp_enabled,
                        current_chest_y,
                        auto_midline_curve,
                        is_manual_mode,
                    )
                    cv2.imshow(window_name, display)

        cv2.destroyAllWindows()

    if len(clicked_points) < 2:
        print("Error: Fewer than 2 midline points specified. Cannot align.")
        return None

    # Step 3: Math - Head Tilt Angle and Geometric Center (Same method as manual)
    angle, cx, cy = compute_midline_tilt(clicked_points, tilt_angle_offset)

    # Step 4: Straighten Scan (Tilt Compensation via warpAffine)
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    straightened = cv2.warpAffine(enhanced_bgr, rotation_matrix, (w, h), borderValue=(0, 0, 0))

    # Step 5: Non-Rigid Midline Distortion / Warping (Straighten Curve across joints)
    if warp_enabled and len(clicked_points) >= 3:
        pts_arr = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)
        rot_joints = cv2.transform(pts_arr, rotation_matrix).reshape(-1, 2)
        straightened, _ = warp_straighten_midline(straightened, rot_joints, cx)

    # Step 6: Chest Guillotine Cut (Silencing arch below cut if detected)
    effective_chest_y = current_chest_y if current_chest_y is not None else h
    if effective_chest_y < h:
        straightened[effective_chest_y:, :] = 0

    straightened_gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)

    # Step 7: Strict Symmetrical Square Crop (Exact same method as manual)
    half_w = min(cx, w - cx, cy, h - cy)

    if half_w < 30:
        print("Error: Line/center point is too close to the boundary to crop a valid scan.")
        return None

    top = cy - half_w
    bot = cy + half_w
    left = cx - half_w
    right = cx + half_w

    crop_gray = straightened_gray[top:bot, left:right]
    crop_bgr = straightened[top:bot, left:right]

    if crop_gray.size == 0:
        print("Error: Crop resulted in empty image. Please center your line better.")
        return None

    # Step 8: Split and Mirror (Same method as manual)
    mid = crop_gray.shape[1] // 2

    left_high_res = crop_gray[:, :mid]
    right_high_res = cv2.flip(crop_gray[:, mid:], 1)

    left_bgr = crop_bgr[:, :mid]
    right_bgr_mirrored = cv2.flip(crop_bgr[:, mid:], 1)

    left_tensor_ready = cv2.resize(left_bgr, target_size, interpolation=cv2.INTER_AREA)
    right_tensor_ready = cv2.resize(right_bgr_mirrored, target_size, interpolation=cv2.INTER_AREA)

    result_meta = meta.copy() if meta else {}
    result_meta["chest_y"] = effective_chest_y
    result_meta["angle"] = angle
    result_meta["cx"] = cx
    result_meta["cy"] = cy
    result_meta["half_w"] = half_w
    result_meta["midline_x"] = auto_midline_curve
    result_meta["raw_clean"] = img_clean.copy()
    result_meta["crop_box"] = (left, top, right, bot)
    result_meta["raw_crop_box"] = (max(0, cx - half_w), max(0, cy - half_w), min(w, cx + half_w), min(h, cy + half_w))
    result_meta["straightened"] = straightened.copy()
    result_meta["clicked_points"] = list(clicked_points)

    return ProcessedResult(
        left_tensor_ready,
        right_tensor_ready,
        left_high_res,
        right_high_res,
        crop_bgr,
        meta=result_meta,
    )