import os
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Import patient text removal to eliminate scanner font shortcuts
try:
    from preprocess import remove_patient_text
except ImportError:
    def remove_patient_text(gray_img):
        return gray_img


# 1. ASPECT-RATIO PRESERVING PREPROCESSING

def letterbox_pad(img_bgr, target_size=(224, 224), pad_color=(0, 0, 0)):
    # Pads an image to a square canvas with constant borders before resizing.
    # Prevents vertical compression of tall FULL_ARCH scans.

    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    
    pad_top = (max_dim - h) // 2
    pad_bot = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    
    padded = cv2.copyMakeBorder(img_bgr, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    return cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)

def preprocess_for_fov(img_bgr, target_size=(224, 224)):
    # Strips outer text watermarks and pads to square aspect ratio.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clean_gray = remove_patient_text(gray)
    clean_bgr = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
    return letterbox_pad(clean_bgr, target_size=target_size)


# 2. MODEL DEFINITION (FROZEN BACKBONE FOR SMALL DATASETS)

def get_fov_model(num_classes=2, pretrained=True, freeze_backbone=True):
    # MobileNetV3-Small with frozen feature extraction layers to prevent
    # catastrophic forgetting on small medical datasets.

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
    return model


# 3. DATASET & AUGMENTATION

class FOVDataset(Dataset):
    def __init__(self, file_paths, labels, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

        self.train_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply artifact stripping and letterbox square padding
        clean_square = preprocess_for_fov(img, target_size=(224, 224))
        rgb = cv2.cvtColor(clean_square, cv2.COLOR_BGR2RGB)

        tensor = self.train_augment(rgb) if self.is_train else self.val_transform(rgb)
        label = self.labels[idx]
        return tensor, torch.tensor(label, dtype=torch.long)


# 4. TRAINING & VALIDATION PIPELINE

def train_fov_classifier(
    data_dir="data/fov_dataset",
    model_save_path="fov_model.pt",
    epochs=15,
    batch_size=8,
    lr=1e-3,
    val_split=0.20
):
    # Trains FOV classifier with linear probing, hold-out validation,
    # and automatic checkpointing of the lowest validation loss.

    classes = ["FULL_ARCH", "HEAD_ONLY"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    file_paths = []
    labels = []
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"[!] Warning: Directory not found: {cls_dir}")
            continue
            
        for f in os.listdir(cls_dir):
            if f.lower().endswith(valid_exts):
                file_paths.append(os.path.join(cls_dir, f))
                labels.append(cls_idx)

    if not file_paths:
        print(f"Aborted: No images found in '{data_dir}'.")
        return

    # Deterministic train/validation split
    combined = list(zip(file_paths, labels))
    random.seed(42)
    random.shuffle(combined)
    
    val_count = max(1, int(len(combined) * val_split))
    val_samples = combined[:val_count]
    train_samples = combined[val_count:]

    train_paths, train_lbls = zip(*train_samples)
    val_paths, val_lbls = zip(*val_samples)

    train_dataset = FOVDataset(train_paths, train_lbls, is_train=True)
    val_dataset = FOVDataset(val_paths, val_lbls, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset: {len(train_paths)} train, {len(val_paths)} validation | Classes: {class_to_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_fov_model(num_classes=len(classes), pretrained=True, freeze_backbone=True).to(device)

    # Optimize only unfrozen head parameters to prevent catastrophic forgetting
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    print(f"\nTraining on {device}...")

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

        epoch_train_loss = train_loss / len(train_paths)
        epoch_train_acc = (train_correct / len(train_paths)) * 100.0

        # Validation loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                val_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

        epoch_val_loss = val_loss / len(val_paths)
        epoch_val_acc = (val_correct / len(val_paths)) * 100.0

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.1f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)

    print(f"\nTraining complete. Best checkpoint saved as '{model_save_path}'.")


# 5. INFERENCE FUNCTION

def classify_fov(image_path_or_bgr, model_weights="fov_model.pt", device=None):
    # Evaluates image and returns predicted FOV label and softmax confidence.

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model checkpoint '{model_weights}' not found. Train model first.")

    model = get_fov_model(num_classes=2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device).eval()

    if isinstance(image_path_or_bgr, str):
        img_bgr = cv2.imread(image_path_or_bgr)
        if img_bgr is None:
            raise ValueError(f"Could not load scan from: {image_path_or_bgr}")
    else:
        img_bgr = image_path_or_bgr

    clean_square = preprocess_for_fov(img_bgr, target_size=(224, 224))
    rgb = cv2.cvtColor(clean_square, cv2.COLOR_BGR2RGB)

    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = normalize(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    class_names = ["FULL_ARCH", "HEAD_ONLY"]
    return class_names[pred_idx], float(probs[pred_idx].item())

if __name__ == "__main__":
    train_fov_classifier(data_dir="data/fov_dataset")
