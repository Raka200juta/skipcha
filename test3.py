#!/usr/bin/env python3
"""
login_with_model.py
Integrasi model (.h5 atau .pth) ke pipeline login yang kamu punya.
Urutan: pakai Keras .h5 (jika ada) -> pakai PyTorch .pth (jika ada) -> fallback Tesseract OCR -> fallback Selenium.
Simpan file ini di folder yang sama dengan model dan file login aslinya.
"""

import os
import time
import re
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import requests
import traceback

# --- Konfigurasi model filenames (ubah sesuai nama filemu) ---
H5_NAME = "lenet.hdf5"   # contoh nama .h5
PTH_NAME = "model.pth"       # contoh nama .pth

# --- Basic login config (salin dari script asli) ---
LOGIN_URL = "https://mlive.minemedia.tv/"
USERNAME = "99988805682"
PASSWORD = "92076958"
GENERATE_CAPTCHA_URL = "https://mlive.minemedia.tv/v3/util/generate_captcha?cptc=2&t="
GET_CAPTCHA_URL = "https://mlive.minemedia.tv/v3/util/get_captcha?cptc=2&id="
LOGIN_API_URL = "https://mlive.minemedia.tv/v3/users/web_login/"

# --- Session ---
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0',
    'Referer': LOGIN_URL
})

# -------------------------
#  Utility: preprocessing
# -------------------------
def preprocess_pil_image_for_model(pil_img, target_size=(28,28)):
    """
    Input: PIL.Image (RGB or L)
    Output:
      - keras: numpy array shape (1,28,28,1), float32 scaled [0,1]
      - pytorch: torch tensor equivalent expected by user code (caller will handle)
    """
    # convert to grayscale
    img = pil_img.convert("L")
    # convert to numpy
    arr = np.array(img)
    # resize
    arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
    # normalize to 0..1
    arr = arr.astype("float32") / 255.0
    # add channel
    arr = np.expand_dims(arr, axis=-1)   # H,W,1
    arr = np.expand_dims(arr, axis=0)    # 1,H,W,1
    return arr

# -------------------------
#  Keras loader & predict
# -------------------------
keras_available = False
keras_model = None
try:
    from tensorflow.keras.models import load_model
    keras_available = True
except Exception:
    keras_available = False

def try_load_keras_model(path=H5_NAME):
    global keras_model
    if not keras_available:
        return False
    if not os.path.exists(path):
        return False
    try:
        keras_model = load_model(path)
        print(f"[MODEL] Loaded Keras model from {path}")
        return True
    except Exception as e:
        print("[MODEL] Failed loading Keras model:", e)
        traceback.print_exc()
        return False

def predict_with_keras(pil_img):
    """
    returns predicted label (string) or None
    """
    if keras_model is None:
        return None
    arr = preprocess_pil_image_for_model(pil_img)  # (1,28,28,1)
    preds = keras_model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    # default mapping: idx -> str(idx) ; ubah jika kamu punya mapping lain
    return str(idx)

# -------------------------
#  PyTorch loader & predict
# -------------------------
pytorch_available = False
torch = None
torch_model = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    pytorch_available = True
except Exception:
    pytorch_available = False

class LeNetTorch(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def try_load_torch_model(path=PTH_NAME, device=None):
    global torch_model, torch
    if not pytorch_available:
        return False
    if not os.path.exists(path):
        return False
    try:
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # instantiate model and load state_dict
        m = LeNetTorch(num_classes=10)
        state = torch.load(path, map_location=device)
        # If state is full model (saved by torch.save(model)), try load directly
        try:
            m.load_state_dict(state)
        except Exception:
            try:
                # sometimes saved as {'model_state_dict': ...}
                if isinstance(state, dict):
                    if 'model_state_dict' in state:
                        m.load_state_dict(state['model_state_dict'])
                    else:
                        # try common keys
                        # try to find first dict of tensors
                        m.load_state_dict(state)
            except Exception as e2:
                print("Torch: couldn't load state_dict:", e2)
                raise
        m.to(device)
        m.eval()
        torch_model = (m, device)
        print(f"[MODEL] Loaded PyTorch model from {path} on {device}")
        return True
    except Exception as e:
        print("[MODEL] Failed loading PyTorch model:", e)
        traceback.print_exc()
        return False

def predict_with_torch(pil_img):
    """
    returns predicted label (string) or None
    """
    if torch_model is None:
        return None
    m, device = torch_model
    # preprocess to tensor: 1x1x28x28 normalized same as training
    arr = pil_img.convert("L")
    arr = np.array(arr).astype("float32")
    arr = cv2.resize(arr, (28,28))
    arr = arr / 255.0
    # to tensor
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    t = t.to(device)
    with torch.no_grad():
        out = m(t)
        pred_idx = int(out.argmax(dim=1).cpu().numpy()[0])
    return str(pred_idx)

# -------------------------
#  Fallback: Tesseract OCR
# -------------------------
try:
    import pytesseract
    tesseract_available = True
except Exception:
    tesseract_available = False

def predict_with_tesseract(pil_img):
    if not tesseract_available:
        return None
    # basic preprocess similar to earlier script
    img = pil_img.convert("L")
    arr = np.array(img)
    _, thr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pil2 = Image.fromarray(thr)
    text = pytesseract.image_to_string(pil2, config="--psm 8 -c tessedit_char_whitelist=0123456789")
    cleaned = re.sub(r'[^0-9]', '', text)
    return cleaned if cleaned else None

# -------------------------
#  API helper (minimal)
# -------------------------
def generate_captcha_id():
    ts = int(time.time()*1000)
    url = f"{GENERATE_CAPTCHA_URL}{ts}"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        captcha_id = (data.get('data') or {}).get('id') or data.get('id') or data.get('captcha_id')
        return captcha_id
    except Exception as e:
        print("generate_captcha_id error:", e)
        return None

def get_captcha_image_by_id(captcha_id):
    if not captcha_id:
        return None
    url = f"{GET_CAPTCHA_URL}{captcha_id}"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception as e:
        print("get_captcha_image_by_id error:", e)
        return None

# -------------------------
#  High-level solve function
# -------------------------
def solve_captcha_with_models(pil_img):
    """
    Try keras -> torch -> tesseract. Return string or None
    """
    # try keras first
    if keras_model is not None:
        try:
            pred = predict_with_keras(pil_img)
            if pred:
                print("[PREDICT] Keras:", pred)
                return pred
        except Exception as e:
            print("Keras predict error:", e)
    # try torch
    if torch_model is not None:
        try:
            pred = predict_with_torch(pil_img)
            if pred:
                print("[PREDICT] PyTorch:", pred)
                return pred
        except Exception as e:
            print("PyTorch predict error:", e)
    # tesseract fallback
    if tesseract_available:
        try:
            pred = predict_with_tesseract(pil_img)
            if pred:
                print("[PREDICT] Tesseract:", pred)
                return pred
        except Exception as e:
            print("Tesseract error:", e)
    return None

# -------------------------
#  Initialize models at startup
# -------------------------
_ = try_load_keras_model(H5_NAME)
_ = try_load_torch_model(PTH_NAME)

# -------------------------
#  Example quick-test function
# -------------------------
def quick_test_from_api():
    cid = generate_captcha_id()
    if not cid:
        print("No captcha id")
        return
    img = get_captcha_image_by_id(cid)
    if not img:
        print("No captcha image")
        return
    img.save("sample_captcha.png")
    print("Saved sample_captcha.png")
    pred = solve_captcha_with_models(img)
    print("Result ->", pred)

if __name__ == "__main__":
    # quick test: download one captcha and try predict
    quick_test_from_api()
