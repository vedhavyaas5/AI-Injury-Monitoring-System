"""
MRI inference service.
Loads the Phase 6 trained model (mri_model.pth) and runs inference
on an uploaded image.

DISCLAIMER: Output is an AI classification result, NOT a medical diagnosis.
"""
import os
import io
import json
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

BASE_DIR   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
META_PATH  = os.path.join(MODELS_DIR, 'mri_model_meta.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'mri_model.pth')

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
MAX_FILE_BYTES   = 10 * 1024 * 1024   # 10 MB

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

DISCLAIMER = (
    "This output is an AI model classification result, NOT a medical diagnosis. "
    "It must NOT be used as the sole basis for any clinical decision. "
    "Always consult a qualified medical professional."
)

_model      = None
_meta       = None
_device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_model():
    global _model, _meta
    if _model is not None:
        return _model, _meta

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'MRI model not found: {MODEL_PATH}')
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f'MRI meta not found: {META_PATH}')

    with open(META_PATH) as f:
        _meta = json.load(f)

    ckpt = torch.load(MODEL_PATH, map_location=_device)

    # Rebuild model architecture
    from torchvision.models import resnet18, ResNet18_Weights
    import torch.nn as nn

    model_name  = ckpt.get('model_name', 'ResNet18')
    num_classes = ckpt.get('num_classes', 2)

    if model_name == 'ResNet18':
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'MobileNetV2':
        from torchvision.models import mobilenet_v2
        m = mobilenet_v2(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif model_name == 'EfficientNet-B0':
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    else:  # SimpleCNN
        from phase6_mri_classification import MODEL_BUILDERS
        m = MODEL_BUILDERS['SimpleCNN'](num_classes)

    m.load_state_dict(ckpt['state_dict'])
    m.to(_device)
    m.eval()
    _model = m
    return _model, _meta


def validate_image_bytes(content: bytes, filename: str) -> None:
    """
    Raise ValueError for invalid uploads.
    Checks: extension, size, readability, corruption.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(
            f'Unsupported file format "{ext}". '
            f'Accepted: {", ".join(VALID_EXTENSIONS)}'
        )
    if len(content) == 0:
        raise ValueError('Uploaded file is empty.')
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(
            f'File too large ({len(content)//1024} KB). Max: {MAX_FILE_BYTES//1024//1024} MB.'
        )
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
    except UnidentifiedImageError:
        raise ValueError('File is not a valid image or is corrupted.')
    except Exception as e:
        raise ValueError(f'Image validation failed: {str(e)}')


def predict_mri(image_bytes: bytes) -> dict:
    """
    Run MRI classification on raw image bytes.

    Returns
    -------
    dict:
        prediction  : str   — class name from trained model
        confidence  : float — softmax probability 0–1
        top_2       : list  — [(class_name, prob_pct), ...]
        model_name  : str
        demo_warning: str   — present if model was trained on pseudo-labels
        disclaimer  : str   — MUST be shown to end-users
    """
    model, meta = _get_model()

    img_size = meta.get('img_size', 224)
    cls_names = meta.get('class_names', ['Normal', 'Abnormal'])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
    ])

    img    = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    top_2    = sorted(
        zip(cls_names, (probs * 100).tolist()),
        key=lambda x: x[1], reverse=True
    )[:2]

    result = {
        'prediction':  cls_names[pred_idx],
        'confidence':  round(float(probs[pred_idx]), 4),
        'confidence_pct': round(float(probs[pred_idx]) * 100, 1),
        'top_2':       [(c, round(p, 1)) for c, p in top_2],
        'model_name':  meta.get('best_model', 'Unknown'),
        'disclaimer':  DISCLAIMER,
    }

    # Warn if demo model
    if 'DEMO' in meta.get('disclaimer', '').upper():
        result['demo_warning'] = (
            'This model was trained on pseudo-labels for code validation only. '
            'Results carry NO clinical meaning. '
            'Retrain with properly labelled MRI data before any real use.'
        )

    return result
