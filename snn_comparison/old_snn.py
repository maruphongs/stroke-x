"""
================================================================================
ANTIGRAVITY STROKE AI - OLD SIAMESE NEURAL NETWORK (BASELINE RECONSTRUCTION)
================================================================================
Reconstructs the original baseline Siamese Neural Network from '.old/best_model.pt'.

Key Characteristics:
  - Backbone: Vanilla ResNet-18 (full feature extractor)
  - Pooling: Single global average pooling at Layer 4 bottleneck (AdaptiveAvgPool2d((1,1)))
  - Hemispheric Fusion: Simple feature vector concatenation (512 + 512 -> 1024 dims)
  - Classification Head: Linear(1024, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3)
                         -> Linear(256, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 1)
  - Normalization: BatchNorm1d in head (sensitive to small clinical batch sizes)
  - Loss: Standard Binary Cross-Entropy (BCEWithLogitsLoss)
================================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

# Resolve path reliably regardless of working directory
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, ".."))
OLD_MODEL_WEIGHTS = os.path.join(_PROJECT_ROOT, ".old", "best_model.pt")


class OldSiameseNet(nn.Module):
    """
    Baseline Siamese Network (ResNet-18 with Global Bottleneck Feature Concatenation).
    Verified 100% state_dict key and tensor dimension match with '.old/best_model.pt'.
    """
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18()
        self.encoder.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, left_img, right_img):
        # Extract 512-D global average pooled feature representation
        feat_l = self.encoder(left_img)
        feat_r = self.encoder(right_img)

        # Concatenate hemispheric features (1024 dims)
        combined = torch.cat([feat_l, feat_r], dim=1)
        logits = self.head(combined).squeeze(1)
        return logits


def load_old_snn_model(weights_path=OLD_MODEL_WEIGHTS, device=None):
    """
    Loads reconstructed OldSiameseNet model and baseline checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(weights_path):
        return None, device

    model = OldSiameseNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict_old_snn(left_img, right_img, model=None, weights_path=OLD_MODEL_WEIGHTS, device=None):
    """
    Runs inference on a bilateral hemisphere pair using the Old SNN baseline.
    Expects (224, 224, 3) BGR or RGB arrays.
    Returns asymmetry probability in range [0.0, 1.0].
    """
    if model is None:
        model, device = load_old_snn_model(weights_path=weights_path, device=device)
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

    l_t = transform(left_img).unsqueeze(0).to(device)
    r_t = transform(right_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(l_t, r_t)
        prob = torch.sigmoid(logits).item()

    return prob
