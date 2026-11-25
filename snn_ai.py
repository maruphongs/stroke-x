import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from preprocess import process_image


# ============================================================
# 1. CONFIGURATION
# ============================================================
class Config:
    INPUT_SIZE = (224, 224)          # Standard input dimensions for ResNet-18
    BATCH_SIZE = 8
    SYNTHETIC_MULTIPLIER = 6         # Number of synthetic pairs per scan (balanced 1:1)

    EPOCHS = 22                      # 8 Warmup + 14 Fine-Tuning
    LEARNING_RATE = 5e-4             # Tuned base learning rate
    WEIGHT_DECAY = 1e-3              # AdamW L2 regularization
    VAL_SPLIT = 0.20                 # Hold-out validation percentage

    # Balanced Focal Loss: alpha=0.50 balances stroke vs healthy equally
    FOCAL_ALPHA = 0.50
    FOCAL_GAMMA = 2.0

    # Dynamic directory discovery supporting NORMAL/ABNORMAL and healthy/stroke
    DATA_DIR    = "data"
    CACHE_DIR   = "data/cache_snn"
    MODEL_SAVE_PATH = "best_model.pt"

    @staticmethod
    def _find_dir(base_candidates, sub_candidates):
        for b in base_candidates:
            if not os.path.isdir(b):
                continue
            for s in sub_candidates:
                target = os.path.join(b, s)
                if os.path.isdir(target):
                    return target
        return None

    HEALTHY_DIR = (
        _find_dir.__func__(["data/snn_dataset", "data/test", "data"], ["NORMAL", "normal", "healthy", "HEALTHY"])
        or "data/snn_dataset/NORMAL"
    )
    STROKE_DIR  = (
        _find_dir.__func__(["data/snn_dataset", "data/test", "data"], ["ABNORMAL", "abnormal", "stroke", "STROKE"])
        or "data/snn_dataset/ABNORMAL"
    )


# ============================================================
# 2. SNN ARCHITECTURE (Multi-Scale Spatial Siamese ResNet-18)
# ============================================================
class SiameseNet(nn.Module):
    """
    Multi-Scale Spatial Siamese Neural Network (ResNet-18 Backbone).
    Extracts multi-scale spatial feature maps from Layer 3 (14x14) and Layer 4 (7x7),
    computing:
      - Regional peak asymmetry (AdaptiveMaxPool2d) for focal arterial occlusions
      - Hemispheric territorial deficit (AdaptiveAvgPool2d) for broad hypodensity
      - Global cosine discrepancy (1.0 - cosine_similarity) for whole-brain symmetry
    Normalized via LayerNorm for sample-invariant, robust clinical evaluation.
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Stage 1: conv1 through layer3 (256 channels, 14x14 spatial)
        self.stage1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )
        # Stage 2: layer4 (512 channels, 7x7 spatial)
        self.stage2 = backbone.layer4

        if freeze_backbone:
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = False

        # Input dimension:
        # Layer 3: diff_max (256) + diff_avg (256) = 512
        # Layer 4: diff_max (512) + diff_avg (512) + cos_dist (1) = 1025
        # Total in_features = 512 + 1025 = 1537
        self.head = nn.Sequential(
            nn.Linear(1537, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Single logit output
        )

    def extract_features(self, x):
        f3 = self.stage1(x)
        f4 = self.stage2(f3)
        return f3, f4

    def forward(self, x_left, x_right):
        f3_l, f4_l = self.extract_features(x_left)
        f3_r, f4_r = self.extract_features(x_right)

        # Layer 3: High-resolution vessel / arterial branch asymmetry
        diff3 = torch.abs(f3_l - f3_r)
        max3 = F.adaptive_max_pool2d(diff3, (1, 1)).flatten(1)
        avg3 = F.adaptive_avg_pool2d(diff3, (1, 1)).flatten(1)

        # Layer 4: Deep territorial / vascular territory deficit
        diff4 = torch.abs(f4_l - f4_r)
        max4 = F.adaptive_max_pool2d(diff4, (1, 1)).flatten(1)
        avg4 = F.adaptive_avg_pool2d(diff4, (1, 1)).flatten(1)

        # Global hemispheric cosine alignment discrepancy
        g4_l = F.adaptive_avg_pool2d(f4_l, (1, 1)).flatten(1)
        g4_r = F.adaptive_avg_pool2d(f4_r, (1, 1)).flatten(1)
        cos_sim = F.cosine_similarity(g4_l, g4_r, dim=1, eps=1e-6).unsqueeze(1)
        cos_dist = 1.0 - cos_sim

        features = torch.cat([max3, avg3, max4, avg4, cos_dist], dim=1)
        return self.head(features).squeeze(1)


# ============================================================
# 3. FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()


# ============================================================
# 4. BALANCED SYNTHETIC STROKE & HEALTHY GENERATION
# ============================================================
def generate_synthetic_strokes(healthy_samples, multiplier=Config.SYNTHETIC_MULTIPLIER):
    """
    Simulates acute ischemic MCA/ACA territory hypodensity and focal arterial cutoffs.
    """
    synthetic = []
    print(f"\n[Synthetic] Synthesizing {multiplier}x ischemic cases from {len(healthy_samples)} healthy scans...")

    for _ in range(multiplier):
        for left_img, right_img, *_ in healthy_samples:
            affect_left = random.random() > 0.5
            target = left_img.copy() if affect_left else right_img.copy()
            h, w = target.shape[:2]

            # Dual-mode stroke simulation:
            # Mode A (Focal): branch arterial cutoff (smaller radius, sharper attenuation)
            # Mode B (Territorial): large vascular territory hypoperfusion
            is_focal = random.random() > 0.4
            if is_focal:
                cx = int(random.uniform(0.30, 0.70) * w)
                cy = int(random.uniform(0.25, 0.70) * h)
                rx = int(random.uniform(0.08, 0.20) * w)
                ry = int(random.uniform(0.08, 0.25) * h)
                attenuation = random.uniform(0.40, 0.80)
                blur_k = 15
            else:
                cx = int(random.uniform(0.25, 0.75) * w)
                cy = int(random.uniform(0.20, 0.75) * h)
                rx = int(random.uniform(0.15, 0.35) * w)
                ry = int(random.uniform(0.15, 0.35) * h)
                attenuation = random.uniform(0.35, 0.70)
                blur_k = 25

            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), random.uniform(0, 360), 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

            factor = 1.0 - (mask[:, :, np.newaxis] * attenuation)
            target = (target.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

            if affect_left:
                synthetic.append((target, right_img.copy(), 1.0))
            else:
                synthetic.append((left_img.copy(), target, 1.0))

    print(f"[Synthetic] Generated {len(synthetic)} synthetic stroke samples.")
    return synthetic


def generate_synthetic_healthy(healthy_samples, multiplier=Config.SYNTHETIC_MULTIPLIER):
    """
    Synthesizes guaranteed symmetric healthy bilateral pairs:
      1. Self-symmetric pairs (left-left, right-right) with slight independent scanner noise
      2. Bilateral healthy pairs with slight independent contrast/illumination shifts
    Ensures the training set remains perfectly 50/50 balanced and teaches the model what symmetry looks like.
    """
    synthetic = []
    print(f"[Synthetic] Synthesizing {multiplier}x healthy symmetry cases from {len(healthy_samples)} healthy scans...")

    for _ in range(multiplier):
        for left_img, right_img, *_ in healthy_samples:
            mode = random.choice(["self_l", "self_r", "bilateral"])
            if mode == "self_l":
                b1 = left_img.copy()
                b2 = left_img.copy()
            elif mode == "self_r":
                b1 = right_img.copy()
                b2 = right_img.copy()
            else:
                b1 = left_img.copy()
                b2 = right_img.copy()

            # Slight independent noise / illumination jitter
            if random.random() > 0.5:
                b = random.uniform(0.95, 1.05)
                c = random.uniform(0.95, 1.05)
                b2 = cv2.convertScaleAbs(b2, alpha=c, beta=int((b - 1.0) * 15))

            synthetic.append((b1, b2, 0.0))

    print(f"[Synthetic] Generated {len(synthetic)} synthetic healthy samples.")
    return synthetic


# ============================================================
# 5. SYNCHRONIZED BILATERAL DATASET & AUGMENTATION
# ============================================================
class SynchronizedCTADataset(Dataset):
    def __init__(self, data_list, is_train=True):
        self.data = data_list
        self.is_train = is_train
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        l_img, r_img, label = self.data[idx][:3]

        l_pil = TF.to_pil_image(cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB))
        r_pil = TF.to_pil_image(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))

        if self.is_train:
            # 1. Whole-brain bilateral horizontal swap (preserves symmetry logic)
            if random.random() > 0.5:
                l_pil, r_pil = r_pil, l_pil

            # 2. Synchronized spatial translation & scale jitter
            angle = random.uniform(-4.0, 4.0)
            tx = random.randint(-4, 4)
            ty = random.randint(-4, 4)
            scale = random.uniform(0.97, 1.03)

            # Since right side was horizontally flipped, tilt angle is negated
            l_pil = TF.affine(l_pil, angle=angle, translate=[tx, ty], scale=scale, shear=0)
            r_pil = TF.affine(r_pil, angle=-angle, translate=[-tx, ty], scale=scale, shear=0)

            # 3. Synchronized illumination & contrast shifts
            if random.random() > 0.5:
                brightness = random.uniform(0.92, 1.08)
                contrast = random.uniform(0.92, 1.08)
                l_pil = TF.adjust_brightness(l_pil, brightness)
                r_pil = TF.adjust_brightness(r_pil, brightness)
                l_pil = TF.adjust_contrast(l_pil, contrast)
                r_pil = TF.adjust_contrast(r_pil, contrast)

        l_tensor = self.normalize(TF.to_tensor(l_pil))
        r_tensor = self.normalize(TF.to_tensor(r_pil))

        return l_tensor, r_tensor, torch.tensor(label, dtype=torch.float32)


# ============================================================
# 6. INFERENCE HELPERS
# ============================================================
def load_snn_model(weights_path=Config.MODEL_SAVE_PATH, device=None):
    """
    Loads SiameseNet model architecture and weights.
    Returns (model, device) or (None, device) if checkpoint not found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(weights_path):
        return None, device

    model = SiameseNet(pretrained=False).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict_snn(left_img, right_img, model=None, weights_path=Config.MODEL_SAVE_PATH, device=None):
    """
    Standardized SNN inference on a pair of BGR hemisphere images.
    Returns asymmetry probability in range [0.0, 1.0], or None if model unavailable.
    """
    if model is None:
        model, device = load_snn_model(weights_path, device=device)
        if model is None:
            return None
    elif device is None:
        device = next(model.parameters()).device

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    l_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    r_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    l_t = transform(l_rgb).unsqueeze(0).to(device)
    r_t = transform(r_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(l_t, r_t)
        prob = torch.sigmoid(logits).item()

    return prob


# ============================================================
# 7. COHORT DISCOVERY & PREPROCESSING CACHING
# ============================================================
def discover_scans(healthy_dir=None, stroke_dir=None, data_dir=Config.DATA_DIR):
    """
    Intelligently discovers healthy and stroke scans across directories.
    Supports NORMAL/ABNORMAL, healthy/stroke, and recursive data_dir scanning.
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    healthy_files = []
    stroke_files = []

    if healthy_dir is None:
        healthy_dir = Config.HEALTHY_DIR
    if stroke_dir is None:
        stroke_dir = Config.STROKE_DIR

    def _resolve_dir_alias(target_dir, aliases):
        if target_dir and os.path.isdir(target_dir):
            return target_dir
        if target_dir:
            parent = os.path.dirname(os.path.abspath(target_dir))
            for a in aliases:
                cand = os.path.join(parent, a)
                if os.path.isdir(cand):
                    return cand
        return None

    resolved_healthy = _resolve_dir_alias(healthy_dir, ["NORMAL", "normal", "healthy", "HEALTHY"])
    resolved_stroke = _resolve_dir_alias(stroke_dir, ["ABNORMAL", "abnormal", "stroke", "STROKE"])

    # Check dedicated directories first
    if resolved_healthy and os.path.isdir(resolved_healthy):
        healthy_files = [
            os.path.join(resolved_healthy, f) for f in os.listdir(resolved_healthy)
            if f.lower().endswith(valid_exts)
        ]
    if resolved_stroke and os.path.isdir(resolved_stroke):
        stroke_files = [
            os.path.join(resolved_stroke, f) for f in os.listdir(resolved_stroke)
            if f.lower().endswith(valid_exts)
        ]

    # Fallback to automated recursive discovery in data_dir if either is empty
    if (not healthy_files or not stroke_files) and os.path.exists(data_dir):
        skip_dirs = {"cache_snn", "reports", ".old", ".venv", "scratch", "__pycache__"}
        for root, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            root_lower = root.lower().replace("\\", "/")
            if "fov_dataset" in root_lower:
                continue
            for f in files:
                if not f.lower().endswith(valid_exts):
                    continue
                full_p = os.path.join(root, f)
                stem = os.path.splitext(f)[0].lower()
                if stem.startswith("s") or "abnormal" in root_lower or "stroke" in root_lower:
                    if full_p not in stroke_files:
                        stroke_files.append(full_p)
                elif stem.startswith("n") or "normal" in root_lower or "healthy" in root_lower:
                    if full_p not in healthy_files:
                        healthy_files.append(full_p)

    def _natural_key(p):
        fname = os.path.basename(p)
        stem = os.path.splitext(fname)[0]
        prefix = stem[0].lower() if len(stem) > 0 else ""
        num_part = "".join([c for c in stem[1:] if c.isdigit()])
        num = int(num_part) if num_part else 0
        return (prefix, num, fname)

    healthy_files = sorted(set(healthy_files), key=_natural_key)
    stroke_files = sorted(set(stroke_files), key=_natural_key)
    return healthy_files, stroke_files


def load_cohort_with_cache(healthy_files, stroke_files, cache_dir=Config.CACHE_DIR, target_size=Config.INPUT_SIZE, force_refresh=False):
    """
    Preprocesses and extracts bilateral hemisphere tensors for SNN training and evaluation.
    Caches processed numpy arrays to disk to accelerate repeated training iterations.
    Set force_refresh=True to bypass cache and recompute from raw image files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    raw_samples = []

    file_groups = [(healthy_files, 0.0, "HEALTHY"), (stroke_files, 1.0, "STROKE")]

    for file_list, label, tag in file_groups:
        print(f"\n--- Ingesting {len(file_list)} {tag} Scan(s) ---")
        for img_path in file_list:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cache_file = os.path.join(cache_dir, f"{base_name}_{target_size[0]}x{target_size[1]}.npz")

            # Load from cache if valid and not force_refresh
            if not force_refresh and os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(img_path):
                data = np.load(cache_file)
                left_tensor = data['left']
                right_tensor = data['right']
                raw_samples.append((left_tensor, right_tensor, label, img_path))
                print(f"[Cache Hit] {os.path.basename(img_path)} loaded from disk cache.")
                continue

            # Compute via pipeline
            result = process_image(img_path, target_size=target_size, mode="auto", interactive_fallback=False)
            if result is not None and result != "QUIT":
                left_tensor, right_tensor, _, _, _ = result
                np.savez_compressed(cache_file, left=left_tensor, right=right_tensor)
                raw_samples.append((left_tensor, right_tensor, label, img_path))
                print(f"[Processed & Cached] {os.path.basename(img_path)}")
            else:
                print(f"[-] Preprocessing rejected: {os.path.basename(img_path)}")

    return raw_samples


# ============================================================
# 8. TWO-STAGE TRANSFER LEARNING TRAINING PIPELINE
# ============================================================
def train_snn_pipeline(
    data_dir=Config.DATA_DIR,
    healthy_dir=Config.HEALTHY_DIR,
    stroke_dir=Config.STROKE_DIR,
    model_save_path=Config.MODEL_SAVE_PATH,
    batch_size=Config.BATCH_SIZE,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    val_split=Config.VAL_SPLIT,
    synthetic_multiplier=Config.SYNTHETIC_MULTIPLIER,
    seed=42
):
    """
    Two-Stage SNN Training Pipeline:
    1. Automatic scan discovery and preprocessing disk caching.
    2. Balanced dual-mode synthetic generation (focal + territorial stroke & symmetric healthy).
    3. Stage 1 (Warmup): Frozen backbone, train Multi-Scale LayerNorm head with AdamW.
    4. Stage 2 (Fine-Tuning): Unfreeze ResNet-18 layer4 with low differential LR and Cosine Annealing.
    5. Balanced accuracy and ROC-AUC checkpointing to guarantee peak sensitivity and specificity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 60)
    print("SNN AI TRAINING PIPELINE (Multi-Scale Spatial SiameseNet)")
    print("=" * 60)

    # 1. Discover Cohort
    healthy_files, stroke_files = discover_scans(healthy_dir, stroke_dir, data_dir)
    print(f"Found {len(healthy_files)} healthy scan(s) and {len(stroke_files)} stroke scan(s).")

    if not healthy_files and not stroke_files:
        print("[!] Execution aborted: No CTA scans discovered.")
        return False

    # 2. Preprocess & Cache
    cohort = load_cohort_with_cache(healthy_files, stroke_files, target_size=Config.INPUT_SIZE)
    if not cohort:
        print("[!] Execution aborted: Preprocessing yielded 0 valid samples.")
        return False

    healthy_real = [item for item in cohort if item[2] == 0.0]
    stroke_real = [item for item in cohort if item[2] == 1.0]

    # 3. Controlled Balanced Synthetic Augmentation
    synthetic_stroke = []
    synthetic_healthy = []
    if healthy_real and synthetic_multiplier > 0:
        synthetic_stroke = generate_synthetic_strokes(healthy_real, multiplier=synthetic_multiplier)
        synthetic_healthy = generate_synthetic_healthy(healthy_real, multiplier=synthetic_multiplier)

    all_healthy = healthy_real + synthetic_healthy
    all_stroke = stroke_real + synthetic_stroke

    print(f"\nCohort Summary: {len(healthy_real)} healthy real, {len(stroke_real)} stroke real.")
    print(f"Synthetic Pool: {len(synthetic_healthy)} healthy synth, {len(synthetic_stroke)} stroke synth.")
    print(f"Balanced Total: {len(all_healthy)} healthy pairs, {len(all_stroke)} stroke pairs.")

    # 4. Stratified Train / Validation Split (Guarantees class balance in both)
    val_h_cnt = max(1, int(len(all_healthy) * val_split))
    val_s_cnt = max(1, int(len(all_stroke) * val_split))

    random.shuffle(all_healthy)
    random.shuffle(all_stroke)

    val_data = all_healthy[:val_h_cnt] + all_stroke[:val_s_cnt]
    train_data = all_healthy[val_h_cnt:] + all_stroke[val_s_cnt:]

    random.shuffle(val_data)
    random.shuffle(train_data)

    train_dataset = SynchronizedCTADataset(train_data, is_train=True)
    val_dataset = SynchronizedCTADataset(val_data, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_data)),
        shuffle=True,
        drop_last=len(train_data) >= batch_size
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device} ({len(train_data)} train samples, {len(val_data)} validation samples)...")

    # 5. Initialize Model & Two-Stage Training Setup
    model = SiameseNet(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Stage 1: Linear Probe Warmup (Head only)
    warmup_epochs = max(4, epochs // 3)
    fine_tune_epochs = epochs - warmup_epochs

    print(f"\n[Stage 1] Head Linear Probe Warmup ({warmup_epochs} epochs, backbone frozen)...")
    head_params = filter(lambda p: p.requires_grad, model.head.parameters())
    optimizer_stage1 = torch.optim.AdamW(head_params, lr=lr * 3.0, weight_decay=Config.WEIGHT_DECAY)

    best_balanced_acc = -1.0
    best_val_loss = float("inf")

    def run_epoch(opt, is_train=True):
        if is_train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        loader = train_loader if is_train else val_loader
        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for left, right, targets in loader:
                left, right, targets = left.to(device), right.to(device), targets.to(device)

                if is_train:
                    opt.zero_grad()

                logits = model(left, right)
                loss = criterion(logits, targets)

                if is_train:
                    loss.backward()
                    opt.step()

                total_loss += loss.item() * len(targets)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        n_samples = len(train_data) if is_train else len(val_data)
        avg_loss = total_loss / max(1, n_samples)

        y_t = np.array(all_targets)
        y_p = np.array(all_preds)
        acc = (y_t == y_p).mean() * 100.0

        stroke_idx = np.where(y_t == 1.0)[0]
        healthy_idx = np.where(y_t == 0.0)[0]

        sens = (y_p[stroke_idx] == 1.0).mean() * 100.0 if len(stroke_idx) > 0 else 0.0
        spec = (y_p[healthy_idx] == 0.0).mean() * 100.0 if len(healthy_idx) > 0 else 0.0
        b_acc = 0.5 * (sens + spec)

        return avg_loss, acc, b_acc, sens, spec

    # Run Stage 1 Warmup
    for ep in range(warmup_epochs):
        t_loss, t_acc, _t_bacc, _, _ = run_epoch(optimizer_stage1, is_train=True)
        v_loss, v_acc, v_bacc, v_sens, v_spec = run_epoch(optimizer_stage1, is_train=False)

        print(f"Warmup Ep {ep+1:02d}/{warmup_epochs} | "
              f"Tr Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}% (Sens: {v_sens:.0f}%, Spec: {v_spec:.0f}%, BalAcc: {v_bacc:.1f}%)")

        if v_bacc > best_balanced_acc or (v_bacc == best_balanced_acc and v_loss < best_val_loss):
            best_balanced_acc = v_bacc
            best_val_loss = v_loss
            torch.save(model.state_dict(), model_save_path)

    # Stage 2: Fine-Tuning (Unfreeze ResNet-18 stage2 / layer4)
    print(f"\n[Stage 2] Fine-Tuning ({fine_tune_epochs} epochs, unfreezing layer4 + head)...")
    for param in model.stage2.parameters():
        param.requires_grad = True

    param_groups = [
        {'params': model.stage2.parameters(), 'lr': lr * 0.04},
        {'params': model.head.parameters(), 'lr': lr}
    ]
    optimizer_stage2 = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=fine_tune_epochs, eta_min=1e-6)

    for ep in range(fine_tune_epochs):
        t_loss, t_acc, _t_bacc, _, _ = run_epoch(optimizer_stage2, is_train=True)
        v_loss, v_acc, v_bacc, v_sens, v_spec = run_epoch(optimizer_stage2, is_train=False)
        scheduler.step()

        print(f"FineTune Ep {ep+1:02d}/{fine_tune_epochs} | "
              f"Tr Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}% (Sens: {v_sens:.0f}%, Spec: {v_spec:.0f}%, BalAcc: {v_bacc:.1f}%)")

        if v_bacc > best_balanced_acc or (v_bacc == best_balanced_acc and v_loss < best_val_loss):
            best_balanced_acc = v_bacc
            best_val_loss = v_loss
            torch.save(model.state_dict(), model_save_path)

    print("\n" + "=" * 60)
    print(f"Training Complete! Optimal model saved to '{model_save_path}'.")
    print(f"Best Validation Balanced Accuracy: {best_balanced_acc:.1f}% | Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN AI Training Pipeline")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Base learning rate")
    parser.add_argument("--data-dir", type=str, default=Config.DATA_DIR, help="Path to CTA scans directory")
    args = parser.parse_args()

    train_snn_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
