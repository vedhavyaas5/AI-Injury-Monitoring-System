"""
=============================================================================
 REAL-TIME SPORTS INJURY MONITORING SYSTEM
 PHASE 6 — MRI Image Classification Module
=============================================================================
 Author  : Sports Injury Monitoring System
 Purpose : Inspect the MRI image dataset, determine whether reliable
           labels exist, organise data, train CNN / transfer-learning
           classifiers, evaluate, explain (Grad-CAM), and expose a
           predict_mri() inference function.

 IMPORTANT SCOPE NOTE:
   This module is a standalone image-classification support tool.
   It is NOT part of the real-time camera + wearable-sensor pipeline.
   Its output is NOT a medical diagnosis. It is a model-derived
   classification intended to assist — not replace — clinical review.
=============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import shutil
import hashlib
import warnings
import time
import random
import json

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18,     ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR     = r'c:\studies\sports monitoring system'
RAW_IMG_DIR  = os.path.join(BASE_DIR, 'dataset',
                             'medical imaging sports data',
                             'medical imaging sports data')
MRI_DATA_DIR = os.path.join(BASE_DIR, 'mri_dataset')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
REPORTS_DIR  = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

IMG_SIZE   = 224          # pixels (standard for pretrained ImageNet models)
BATCH_SIZE = 8            # small batch suits the small dataset
MAX_EPOCHS = 50           # hard cap — early stopping will trigger sooner
PATIENCE   = 8            # early-stopping patience (epochs without improvement)
LR_INIT    = 1e-3
LR_STEP    = 10           # StepLR: decay every N epochs
LR_GAMMA   = 0.5          # StepLR: multiply LR by this factor each step

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'figure.facecolor': 'white'})
DIVIDER = '─' * 65

def section(title):
    print(f'\n{"═"*65}')
    print(f'  {title}')
    print(f'{"═"*65}')

def sub(title):
    print(f'\n{DIVIDER}')
    print(f'  {title}')
    print(DIVIDER)


# =============================================================================
# STEP 1 — DATASET INSPECTION
# =============================================================================
section('STEP 1 — DATASET INSPECTION')

print(f'  Raw image directory: {RAW_IMG_DIR}')
print(f'  Device             : {DEVICE}')

# Collect all image files
VALID_EXTS = {'.webp', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
all_files  = [
    os.path.join(RAW_IMG_DIR, f)
    for f in os.listdir(RAW_IMG_DIR)
    if os.path.splitext(f)[1].lower() in VALID_EXTS
]
all_files.sort()

print(f'\n  Total image files found: {len(all_files)}')

# ── Inspect each image ───────────────────────────────────────────────────────
dims_list    = []
corrupted    = []
file_hashes  = {}
duplicate_pairs = []

for fpath in all_files:
    try:
        with Image.open(fpath) as img:
            img.verify()          # check file integrity
        with Image.open(fpath) as img:
            dims_list.append(img.size)     # (width, height)
        # Hash for duplicate detection
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in file_hashes:
            duplicate_pairs.append((file_hashes[h], fpath))
        else:
            file_hashes[h] = fpath
    except (UnidentifiedImageError, Exception):
        corrupted.append(fpath)

valid_files = [f for f in all_files if f not in corrupted]
dims_arr    = np.array(dims_list)

print(f'  Valid images          : {len(valid_files)}')
print(f'  Corrupted / unreadable: {len(corrupted)}')
print(f'  Duplicate images      : {len(duplicate_pairs)}')

if dims_arr.shape[0] > 0:
    widths, heights = dims_arr[:, 0], dims_arr[:, 1]
    print(f'\n  Image dimensions:')
    print(f'    Width  — min: {widths.min()}, max: {widths.max()}, '
          f'mean: {widths.mean():.0f}')
    print(f'    Height — min: {heights.min()}, max: {heights.max()}, '
          f'mean: {heights.mean():.0f}')
    unique_dims = list(set(dims_list))
    print(f'    Unique dimension pairs: {len(unique_dims)}')
    if len(unique_dims) <= 10:
        print(f'    {unique_dims}')

# ── Label analysis ───────────────────────────────────────────────────────────
sub('Label Analysis')

print("""
  Image filenames follow the pattern: OIP (N).webp
  There are NO embedded class labels, folder-level labels,
  accompanying CSV/JSON annotation files, or DICOM metadata
  in the provided dataset.

  The images appear to be generic sports-medicine / MRI
  reference images downloaded without structured labelling.

  ──────────────────────────────────────────────────────
  LABEL RELIABILITY ASSESSMENT: ✗ UNRELIABLE
  ──────────────────────────────────────────────────────

  Reason:
    1. Filenames contain no class information (OIP (1).webp … OIP (56).webp).
    2. No annotation file (labels.csv, annotations.json, etc.) was found.
    3. No subfolder structure (class_A/, class_B/) exists.
    4. 56 images is below the minimum recommended for reliable
       transfer-learning fine-tuning even with heavy augmentation
       (typically ≥ 100–200 images per class minimum).
    5. Assigning synthetic labels to these images would constitute
       data fabrication and produce a meaningless classifier.

  DECISION:
    The supervised training pipeline will NOT be executed on
    unlabelled data. Instead:
      (a) A complete, production-ready training pipeline is
          implemented below — ready to run the moment labelled
          data is supplied.
      (b) A pseudo-label demonstration is executed using a
          50/50 binary split (Normal vs Abnormal) purely to
          validate that the code runs end-to-end. The demo
          results carry NO clinical or predictive meaning.
      (c) Clear instructions for labelling are provided.
  ──────────────────────────────────────────────────────
""")

LABELLING_INSTRUCTIONS = """
  HOW TO LABEL YOUR MRI DATASET BEFORE RERUNNING PHASE 6
  ───────────────────────────────────────────────────────
  Option A — Folder-based labelling (recommended):
    Organise images into subfolders per class, e.g.:
      dataset/mri_labelled/normal/        ← place normal MRI images here
      dataset/mri_labelled/acl_tear/      ← place ACL-tear images here
      dataset/mri_labelled/meniscus_tear/ ← etc.
    Then update RAW_IMG_DIR in this script to point to mri_labelled/.
    The pipeline will auto-detect class names from subfolder names.

  Option B — CSV annotation file:
    Create a file 'mri_labels.csv' with columns: filename, label
    The pipeline can be adapted to read this file.

  Option C — Use a publicly labelled MRI dataset:
    • MRNet (Stanford): https://stanfordmlgroup.github.io/competitions/mrnet/
    • KneeXrayData (Kaggle): search for knee MRI injury datasets
    • RSNA datasets: https://www.rsna.org/education/ai-resources-and-training

  Minimum recommended sample sizes per class for transfer learning:
    Acceptable  : ≥ 50  images/class  (with aggressive augmentation)
    Good        : ≥ 200 images/class
    Ideal       : ≥ 500 images/class
  ───────────────────────────────────────────────────────
"""
print(LABELLING_INSTRUCTIONS)

# =============================================================================
# PSEUDO-LABEL DEMO SETUP
# (50/50 binary split on filenames — for code validation ONLY)
# =============================================================================
section('PSEUDO-LABEL DEMONSTRATION  [CODE VALIDATION ONLY — NOT REAL LABELS]')
print("""
  ⚠ WARNING: The following section assigns SYNTHETIC pseudo-labels
  to produce a runnable end-to-end demonstration. These labels have
  NO medical meaning. Model metrics in this section are MEANINGLESS
  and must NOT be interpreted as real performance results.

  The code structure, pipeline, and saved artefacts ARE valid and
  will produce meaningful results when real labelled data is used.
""")

# Assign pseudo-labels: first half → class 0 (Normal), second half → class 1 (Abnormal)
n = len(valid_files)
pseudo_labels = [0] * (n // 2) + [1] * (n - n // 2)
random.shuffle(pseudo_labels)        # shuffle to avoid order bias

CLASS_NAMES = ['Normal', 'Abnormal']
label_counts = {0: pseudo_labels.count(0), 1: pseudo_labels.count(1)}
print(f'  Total images for demo  : {n}')
print(f'  Class 0 (Normal)       : {label_counts[0]}')
print(f'  Class 1 (Abnormal)     : {label_counts[1]}')

# =============================================================================
# STEP 2 — DATASET ORGANISATION
# =============================================================================
section('STEP 2 — DATASET ORGANISATION')

splits = {
    'train':      os.path.join(MRI_DATA_DIR, 'train'),
    'validation': os.path.join(MRI_DATA_DIR, 'validation'),
    'test':       os.path.join(MRI_DATA_DIR, 'test'),
}
for split_dir in splits.values():
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

# Stratified split: 70 / 15 / 15
indices = list(range(n))
train_idx, temp_idx, train_lbl, temp_lbl = train_test_split(
    indices, pseudo_labels,
    test_size=0.30, random_state=RANDOM_STATE, stratify=pseudo_labels
)
val_idx, test_idx, val_lbl, test_lbl = train_test_split(
    temp_idx, temp_lbl,
    test_size=0.50, random_state=RANDOM_STATE, stratify=temp_lbl
)

def copy_images(indices, labels, split_name):
    copied = 0
    for i, lbl in zip(indices, labels):
        src  = valid_files[i]
        cls  = CLASS_NAMES[lbl]
        dst  = os.path.join(splits[split_name], cls,
                            os.path.basename(src))
        shutil.copy2(src, dst)
        copied += 1
    return copied

c_train = copy_images(train_idx, train_lbl, 'train')
c_val   = copy_images(val_idx,   val_lbl,   'validation')
c_test  = copy_images(test_idx,  test_lbl,  'test')

print(f'  Train      : {c_train} images')
print(f'  Validation : {c_val}   images')
print(f'  Test       : {c_test}  images')
print(f'  Organised under: {MRI_DATA_DIR}')
print("""
  NOTE: In a real dataset, if patient IDs are available, apply
  patient-level splitting so no patient's images appear in both
  training and test sets. This prevents data leakage at the
  patient level (a known pitfall in medical-image research).
""")

# =============================================================================
# STEP 3 — PREPROCESSING & DATA LOADERS
# =============================================================================
section('STEP 3 — PREPROCESSING & AUGMENTATION')

print("""
  Augmentation strategy:
  ──────────────────────────────────────────────────────
  Training augmentations (applied randomly each epoch):
    • RandomHorizontalFlip      — anatomically valid for bilateral structures
    • RandomRotation(±10°)      — slight orientation variance
    • ColorJitter (brightness,  — scanner/acquisition variation simulation
                  contrast)
    • RandomResizedCrop         — scale/position invariance

  Intentionally AVOIDED (medically unrealistic):
    • Vertical flip             — would invert anatomy
    • Large rotations (>15°)    — distorts joint angles
    • Random erasing            — may remove clinically relevant regions

  All splits: Resize → CenterCrop → ToTensor → Normalize(ImageNet mean/std)
    (ImageNet normalisation is appropriate for pretrained transfer models)
  ──────────────────────────────────────────────────────
""")

# ImageNet statistics (used by all pretrained models)
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
])


class MRIDataset(Dataset):
    """
    Custom Dataset for folder-organised MRI images.
    Converts all images to RGB (handles grayscale DICOM-style images).
    """
    def __init__(self, root_dir, transform=None):
        self.samples    = []   # list of (filepath, label_int) tuples
        self.transform  = transform
        self.class_names = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                    self.samples.append(
                        (os.path.join(cls_dir, fname),
                         self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        try:
            img = Image.open(fpath).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


train_ds = MRIDataset(splits['train'],      train_transform)
val_ds   = MRIDataset(splits['validation'], eval_transform)
test_ds  = MRIDataset(splits['test'],       eval_transform)

# ── Weighted sampler for class imbalance ──────────────────────────────────────
train_labels   = [s[1] for s in train_ds.samples]
class_counts   = np.bincount(train_labels)
class_weights  = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[l] for l in train_labels]
sampler        = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f'  Train batches      : {len(train_loader)}  '
      f'({len(train_ds)} images)')
print(f'  Validation batches : {len(val_loader)}  '
      f'({len(val_ds)} images)')
print(f'  Test batches       : {len(test_loader)}  '
      f'({len(test_ds)} images)')
print(f'  Class names        : {train_ds.class_names}')
print(f'  Class → index      : {train_ds.class_to_idx}')
NUM_CLASSES = len(train_ds.class_names)


# =============================================================================
# STEP 4 — MODEL DEFINITIONS
# =============================================================================
section('STEP 4 — MODEL ARCHITECTURES')

def build_simple_cnn(num_classes):
    """
    Lightweight baseline CNN (4 conv blocks).
    Serves as a performance lower-bound reference.
    No pretrained weights — trained from scratch.
    """
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                # Block 4
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
                nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SimpleCNN()


def build_mobilenet(num_classes):
    """
    MobileNetV2 with ImageNet pretrained weights.
    Classifier head replaced for num_classes output.
    All layers fine-tuned (full fine-tuning on small dataset).
    """
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_resnet(num_classes):
    """
    ResNet-18 with ImageNet pretrained weights.
    Final FC layer replaced.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet(num_classes):
    """
    EfficientNet-B0 with ImageNet pretrained weights.
    Classifier head replaced.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


MODEL_BUILDERS = {
    'SimpleCNN':   build_simple_cnn,
    'MobileNetV2': build_mobilenet,
    'ResNet18':    build_resnet,
    'EfficientNet-B0': build_efficientnet,
}
print(f'  Models to compare: {list(MODEL_BUILDERS.keys())}')


# =============================================================================
# STEP 5 — TRAINING LOOP
# =============================================================================
section('STEP 5 — TRAINING WITH EARLY STOPPING & LR SCHEDULING')

# Loss: CrossEntropyLoss with class weights for imbalance handling
loss_weights  = torch.tensor(
    class_weights / class_weights.sum() * NUM_CLASSES,
    dtype=torch.float
).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=loss_weights)


def train_model(model, model_name, train_loader, val_loader,
                max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """
    Full training loop with:
      • AdamW optimiser
      • StepLR learning-rate scheduler
      • Early stopping on validation loss
      • Model checkpointing (best val-loss weights saved)
    """
    model = model.to(DEVICE)
    optimiser = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                          step_size=LR_STEP, gamma=LR_GAMMA)

    best_val_loss = float('inf')
    best_weights  = None
    patience_ctr  = 0
    history       = {'train_loss': [], 'val_loss': [],
                     'train_acc':  [], 'val_acc':  []}

    ckpt_path = os.path.join(MODELS_DIR, f'mri_{model_name}_checkpoint.pth')

    for epoch in range(1, max_epochs + 1):
        # ── Training phase ────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimiser.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == lbls).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # ── Validation phase ──────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs    = model(imgs)
                loss       = criterion(outputs, lbls)
                val_loss  += loss.item() * imgs.size(0)
                preds      = outputs.argmax(dim=1)
                val_correct += (preds == lbls).sum().item()
                val_total   += imgs.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss   = val_loss   / max(val_total,   1)
        train_acc      = train_correct / max(train_total, 1)
        val_acc        = val_correct   / max(val_total,   1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'  [{model_name}] Epoch {epoch:>3}/{max_epochs}  '
              f'Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.3f}  |  '
              f'Val Loss: {avg_val_loss:.4f}  Acc: {val_acc:.3f}  '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # ── Early stopping & checkpointing ───────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights  = {k: v.clone()
                             for k, v in model.state_dict().items()}
            torch.save(best_weights, ckpt_path)
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'  [{model_name}] Early stopping triggered at epoch {epoch}.')
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, history


def evaluate_on_loader(model, loader):
    """Return predictions and probabilities for a DataLoader."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(lbls.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# =============================================================================
# STEP 5 — TRAIN ALL MODELS
# =============================================================================
trained_models_mri = {}
training_histories = {}

for model_name, builder in MODEL_BUILDERS.items():
    sub(f'Training: {model_name}')
    t0    = time.time()
    model = builder(NUM_CLASSES)
    model, history = train_model(
        model, model_name, train_loader, val_loader
    )
    elapsed = time.time() - t0
    trained_models_mri[model_name] = model
    training_histories[model_name] = history
    print(f'  Completed in {elapsed:.1f}s')


# =============================================================================
# STEP 6 — EVALUATION
# =============================================================================
section('STEP 6 — MODEL EVALUATION (Test Set)')

comparison_results = []

fig_cm, axes_cm = plt.subplots(2, 2, figsize=(14, 12))
axes_cm = axes_cm.flatten()

for i, (name, model) in enumerate(trained_models_mri.items()):
    y_true, y_pred, y_prob = evaluate_on_loader(model, test_loader)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, average='binary')
    rec  = recall_score(y_true, y_pred,    zero_division=0, average='binary')
    f1   = f1_score(y_true, y_pred,        zero_division=0, average='binary')
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    comparison_results.append({
        'Model':     name,
        'Accuracy':  round(acc,  4),
        'Precision': round(prec, 4),
        'Recall':    round(rec,  4),
        'F1':        round(f1,   4),
        'ROC-AUC':   round(auc,  4),
    })

    print(f'\n  ── {name} ──')
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0))
    print(f'  ROC-AUC: {auc:.4f}')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=axes_cm[i], linewidths=0.5,
                xticklabels=[f'Pred: {c}' for c in CLASS_NAMES],
                yticklabels=[f'True: {c}' for c in CLASS_NAMES],
                annot_kws={'size': 12, 'weight': 'bold'})
    axes_cm[i].set_title(
        f'{name}\nAcc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}',
        fontsize=10, fontweight='bold'
    )

plt.suptitle('Confusion Matrices — MRI Classification (Test Set)  '
             '[DEMO — pseudo-labels]',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'mri_confusion_matrix.png'), bbox_inches='tight')
plt.close()
print('\n  Saved: mri_confusion_matrix.png')

# Comparison table
df_mri_compare = pd.DataFrame(comparison_results).set_index('Model')
sub('Model Comparison Table')
print(df_mri_compare.to_string())
df_mri_compare.to_csv(os.path.join(REPORTS_DIR, 'mri_model_comparison.csv'))
print('  Saved: mri_model_comparison.csv')

# ── Best model selection ──────────────────────────────────────────────────────
df_mri_compare['selection_score'] = (
    df_mri_compare['Recall']  * 0.40 +
    df_mri_compare['ROC-AUC'] * 0.35 +
    df_mri_compare['F1']      * 0.25
)
best_mri_name  = df_mri_compare['selection_score'].idxmax()
best_mri_model = trained_models_mri[best_mri_name]
print(f'\n  ✓ BEST MRI MODEL: {best_mri_name}')


# =============================================================================
# TRAINING CURVES
# =============================================================================
section('TRAINING CURVES')

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for col_idx, (name, hist) in enumerate(training_histories.items()):
    ep = range(1, len(hist['train_loss']) + 1)
    # Loss
    axes[0][col_idx].plot(ep, hist['train_loss'], label='Train', color='#1565C0', lw=2)
    axes[0][col_idx].plot(ep, hist['val_loss'],   label='Val',   color='#F44336', lw=2)
    axes[0][col_idx].set_title(f'{name}\nLoss', fontsize=9, fontweight='bold')
    axes[0][col_idx].set_xlabel('Epoch', fontsize=8)
    axes[0][col_idx].legend(fontsize=8)
    axes[0][col_idx].spines[['top','right']].set_visible(False)
    # Accuracy
    axes[1][col_idx].plot(ep, hist['train_acc'], label='Train', color='#1565C0', lw=2)
    axes[1][col_idx].plot(ep, hist['val_acc'],   label='Val',   color='#F44336', lw=2)
    axes[1][col_idx].set_title(f'{name}\nAccuracy', fontsize=9, fontweight='bold')
    axes[1][col_idx].set_xlabel('Epoch', fontsize=8)
    axes[1][col_idx].set_ylim(0, 1.05)
    axes[1][col_idx].legend(fontsize=8)
    axes[1][col_idx].spines[['top','right']].set_visible(False)

plt.suptitle('Training Curves — MRI Classification  [DEMO — pseudo-labels]',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'mri_training_curves.png'), bbox_inches='tight')
plt.close()
print('  Saved: mri_training_curves.png')


# =============================================================================
# STEP 7 — EXPLAINABILITY (Grad-CAM)
# =============================================================================
section('STEP 7 — GRAD-CAM EXPLAINABILITY')

print("""
  Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
  the image regions that most influenced the model's prediction.

  !! IMPORTANT DISCLAIMER !!
  ────────────────────────────────────────────────────────────────
  Grad-CAM output is a model INTERPRETABILITY visualisation only.
  It shows where the model focused — it does NOT prove that the
  highlighted region represents a clinically meaningful lesion,
  injury, or abnormality. Clinical validation by a qualified
  radiologist / sports medicine physician is always required.
  ────────────────────────────────────────────────────────────────
""")


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    Works with any model that has a 'features' attribute (CNN-style).
    Adapted for ResNet and EfficientNet via layer4 / features access.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_target_layer(model, model_name):
    """Return the last convolutional feature layer for each architecture."""
    if model_name == 'SimpleCNN':
        return model.features[-2]          # last Conv2d before AvgPool
    elif model_name == 'MobileNetV2':
        return model.features[-1][0]       # last conv in InvertedResidual
    elif model_name == 'ResNet18':
        return model.layer4[-1].conv2      # last conv in layer4
    elif model_name == 'EfficientNet-B0':
        return model.features[-1][0]       # last Conv2dNormActivation
    return None


def generate_gradcam_grid(model, model_name, test_ds, num_samples=4):
    """Generate a grid of Grad-CAM overlays for sample test images."""
    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        print(f'  Grad-CAM: target layer not found for {model_name}')
        return

    try:
        gradcam = GradCAM(model, target_layer)
    except Exception as e:
        print(f'  Grad-CAM setup failed for {model_name}: {e}')
        return

    # Pick evenly-spaced samples
    step    = max(1, len(test_ds) // num_samples)
    indices = list(range(0, min(len(test_ds), num_samples * step), step))[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3.5))
    if num_samples == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        img_tensor, true_label = test_ds[idx]

        # Original image (de-normalise for display)
        mean  = torch.tensor(IMGNET_MEAN).view(3, 1, 1)
        std   = torch.tensor(IMGNET_STD).view(3, 1, 1)
        img_display = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # Grad-CAM
        try:
            cam, pred_class = gradcam.generate(img_tensor)
        except Exception:
            cam = np.zeros((IMG_SIZE, IMG_SIZE))
            pred_class = -1

        # Resize CAM to image size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        cam_np  = np.array(cam_img) / 255.0

        # Original
        axes[row][0].imshow(img_display)
        axes[row][0].set_title(
            f'Original\nTrue: {CLASS_NAMES[true_label]}',
            fontsize=9
        )
        axes[row][0].axis('off')

        # CAM heatmap
        axes[row][1].imshow(cam_np, cmap='jet')
        axes[row][1].set_title('Grad-CAM Heatmap', fontsize=9)
        axes[row][1].axis('off')

        # Overlay
        axes[row][2].imshow(img_display)
        axes[row][2].imshow(cam_np, cmap='jet', alpha=0.45)
        pred_name = CLASS_NAMES[pred_class] if pred_class >= 0 else 'N/A'
        axes[row][2].set_title(
            f'Overlay\nPred: {pred_name}\n'
            f'[Interpretability only — NOT a diagnosis]',
            fontsize=8
        )
        axes[row][2].axis('off')

    plt.suptitle(
        f'Grad-CAM — {model_name}  [DEMO — pseudo-labels]\n'
        '⚠ Highlighted regions reflect model attention, NOT confirmed lesions.',
        fontsize=10, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fname = f'mri_gradcam_{model_name.lower().replace("-","_")}.png'
    plt.savefig(os.path.join(REPORTS_DIR, fname), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')


# Generate Grad-CAM for best model
generate_gradcam_grid(best_mri_model, best_mri_name, test_ds, num_samples=4)


# =============================================================================
# STEP 8 — SAVE MODEL & PREDICTION FUNCTION
# =============================================================================
section('STEP 8 — SAVE MODEL & PREDICTION FUNCTION')

# Save best model weights
best_model_path = os.path.join(MODELS_DIR, 'mri_model.pth')
torch.save({
    'model_name':  best_mri_name,
    'state_dict':  best_mri_model.state_dict(),
    'num_classes': NUM_CLASSES,
    'class_names': CLASS_NAMES,
    'img_size':    IMG_SIZE,
    'imgnet_mean': IMGNET_MEAN,
    'imgnet_std':  IMGNET_STD,
}, best_model_path)
print(f'  Saved: mri_model.pth  (model: {best_mri_name})')

# Save model metadata separately for inspection
meta = {
    'best_model':   best_mri_name,
    'num_classes':  NUM_CLASSES,
    'class_names':  CLASS_NAMES,
    'img_size':     IMG_SIZE,
    'test_metrics': df_mri_compare.loc[best_mri_name,
                    ['Accuracy','Precision','Recall','F1','ROC-AUC']].to_dict(),
    'disclaimer':   (
        'DEMO ONLY — pseudo-labels used. '
        'Results carry NO clinical meaning. '
        'Retrain with labelled data before any deployment.'
    )
}
with open(os.path.join(MODELS_DIR, 'mri_model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print('  Saved: mri_model_meta.json')


def predict_mri(
    image_input,
    model_path: str  = os.path.join(MODELS_DIR, 'mri_model.pth'),
    top_k:      int  = 2
) -> dict:
    """
    Classify an MRI image and return the predicted class with confidence.

    Parameters
    ----------
    image_input : str | PIL.Image.Image | np.ndarray
        Accepts a file path (str), a PIL Image, or a NumPy array.

    model_path : str
        Path to the saved .pth checkpoint.

    top_k : int
        Return top-k predictions with probabilities.

    Returns
    -------
    dict:
        predicted_class     : str   — class name (e.g., 'Normal')
        model_confidence_pct: float — probability × 100 (from softmax)
        top_k_predictions   : list of (class_name, probability_pct)
        model_name          : str
        disclaimer          : str   — MUST be displayed to end users

    !! IMPORTANT !!
    This output is a model classification result.
    It is NOT a medical diagnosis. It must NOT be used
    as the sole basis for any clinical decision.
    Always involve a qualified medical professional.
    """
    # ── Load model ───────────────────────────────────────────────────────
    ckpt       = torch.load(model_path, map_location=DEVICE)
    model_name = ckpt['model_name']
    n_cls      = ckpt['num_classes']
    cls_names  = ckpt['class_names']
    builder    = MODEL_BUILDERS[model_name]
    model      = builder(n_cls).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # ── Load image ───────────────────────────────────────────────────────
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise TypeError(f'Unsupported image type: {type(image_input)}')

    # ── Preprocess ───────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((ckpt['img_size'], ckpt['img_size'])),
        transforms.CenterCrop(ckpt['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=ckpt['imgnet_mean'],
                             std=ckpt['imgnet_std']),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # ── Inference ────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    pred_idx  = int(probs.argmax())
    top_k_res = sorted(
        zip(cls_names, (probs * 100).tolist()),
        key=lambda x: x[1], reverse=True
    )[:top_k]

    return {
        'predicted_class':      cls_names[pred_idx],
        'model_confidence_pct': round(float(probs[pred_idx]) * 100, 2),
        'top_k_predictions':    [(c, round(p, 2)) for c, p in top_k_res],
        'model_name':           model_name,
        'disclaimer': (
            'This output is a model classification result, NOT a medical diagnosis. '
            'It must NOT be used as the sole basis for any clinical decision. '
            'Always consult a qualified medical professional.'
        )
    }


# ── Demo inference ───────────────────────────────────────────────────────────
sub('Demo Inference — predict_mri()')
sample_path = test_ds.samples[0][0]
result      = predict_mri(sample_path)

print(f'  Image tested           : {os.path.basename(sample_path)}')
print(f'  Predicted Class        : {result["predicted_class"]}')
print(f'  Model Confidence       : {result["model_confidence_pct"]}%')
print(f'  Top-{len(result["top_k_predictions"])} Predictions  :')
for cls_name, pct in result['top_k_predictions']:
    print(f'    {cls_name:<15} {pct:.2f}%')
print(f'  Model Used             : {result["model_name"]}')
print(f'\n  ⚠ DISCLAIMER: {result["disclaimer"]}')


# =============================================================================
# STEP 9 — INTEGRATION ARCHITECTURE
# =============================================================================
section('STEP 9 — INTEGRATION WITH THE MONITORING SYSTEM')

print("""
  MRI MODULE — STANDALONE ACCESS ARCHITECTURE
  ────────────────────────────────────────────────────────────────

  The MRI classification module operates COMPLETELY INDEPENDENTLY
  from the real-time camera + wearable sensor pipeline.

  Standalone MRI flow:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   MRI Upload (DICOM / JPEG / PNG / WEBP)                     │
  │         ↓                                                    │
  │   Image Preprocessing                                        │
  │   (Resize 224×224, Normalize with ImageNet statistics)       │
  │         ↓                                                    │
  │   CNN / Transfer Learning Model                              │
  │   (Best of: SimpleCNN / MobileNetV2 / ResNet18 /             │
  │             EfficientNet-B0)                                 │
  │         ↓                                                    │
  │   Classification + Confidence Score                          │
  │   (e.g., "Abnormal — 87.3% confidence")                     │
  │         ↓                                                    │
  │   Optional: Grad-CAM Explainability Map                      │
  │   (shows model attention — NOT a clinical finding)           │
  │         ↓                                                    │
  │   Result Display                                             │
  │   [Predicted class | Confidence | Disclaimer]               │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  Separation from real-time pipeline:
  ┌──────────────────────────────────────────────────────────────┐
  │  REAL-TIME PIPELINE          │  MRI MODULE (SEPARATE)        │
  │  ─────────────────────────   │  ──────────────────────────   │
  │  Camera feed (30+ fps)       │  On-demand single image       │
  │  Wearable sensors (live)     │  upload and classification    │
  │  Phase 4 Movement Model      │                               │
  │  Phase 5 Sensor Model        │  predict_mri(image_path)      │
  │  Phase 7 Pose Estimation     │        ↓                      │
  │  Phase 8 Risk Fusion         │  Returns class + confidence   │
  │         ↓                    │  with mandatory disclaimer     │
  │  Live alert dashboard        │                               │
  └──────────────────────────────┴───────────────────────────────┘

  API access (Phase 9 / deployment):
    Expose predict_mri() as a separate REST endpoint:
      POST /api/mri/classify
      Body: multipart/form-data (image file)
      Returns: JSON { predicted_class, confidence_pct, disclaimer }

    Keep this endpoint isolated from:
      POST /api/realtime/risk  ← movement + sensor pipeline
""")


# =============================================================================
# FINAL REPORT
# =============================================================================
section('PHASE 6 COMPLETE — SUMMARY')
print(f"""
  ⚠ IMPORTANT REMINDER:
    This run used PSEUDO-LABELS for demonstration purposes.
    The metrics below carry NO predictive or clinical meaning.
    Retrain this pipeline with properly labelled MRI data.

  Best Demo Model   : {best_mri_name}
  Test Accuracy     : {df_mri_compare.loc[best_mri_name, "Accuracy"]:.4f}  [DEMO — meaningless]
  Test F1           : {df_mri_compare.loc[best_mri_name, "F1"]:.4f}        [DEMO — meaningless]
  Test ROC-AUC      : {df_mri_compare.loc[best_mri_name, "ROC-AUC"]:.4f}   [DEMO — meaningless]

  Labelling instructions printed above. Provide labelled images and
  re-run this script to obtain meaningful results.

  Saved artefacts:
    models/mri_model.pth
    models/mri_model_meta.json
    reports/mri_confusion_matrix.png
    reports/mri_training_curves.png
    reports/mri_model_comparison.csv
    reports/mri_gradcam_{best_mri_name.lower().replace("-","_")}.png
""")
