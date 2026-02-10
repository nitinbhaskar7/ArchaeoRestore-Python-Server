import json
import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import regex as re

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image, ImageDraw

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL = "models/yolo_det_new.pt"
CLF_MODEL = "models/mobilenet_brahmi_classifier_new.pt"
TRANSFORMER_MODEL = "models/transformer.pth"
VOCAB_PATH = "vocab.json"

TMP_IMAGE = "outputs/input.jpg"
MASKED_IMAGE = "outputs/masked.jpg"
DET_JSON = "outputs/detection_results.json"
CLF_JSON = "outputs/classification_results.json"

CONTEXT_SIZE = 5
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
WINDOW_SIZE = (CONTEXT_SIZE * 2) + 1

os.makedirs("outputs", exist_ok=True)

# =========================
# FASTAPI INIT
# =========================

app = FastAPI(title="Brahmi OCR Missing Character API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODELS (ONCE)
# =========================

# YOLO
model_det = YOLO(YOLO_MODEL)

# Classifier
ckpt = torch.load(CLF_MODEL, map_location=DEVICE)
classes = ckpt["classes"]

model_clf = mobilenet_v2(weights=None)
model_clf.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model_clf.last_channel, len(classes))
)
model_clf.load_state_dict(ckpt["model_state"])
model_clf.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# TRANSFORMER
# =========================

class BrahmiTokenizer:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.unk_idx = self.char2idx.get("<unk>", 2)

    def get_graphemes(self, text):
        return re.findall(r"\X", text)

    def encode(self, g):
        return [self.char2idx.get(x, self.unk_idx) for x in g]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_encoder = nn.Parameter(torch.randn(1, WINDOW_SIZE, EMBED_DIM))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=N_HEADS,
                batch_first=True
            ),
            num_layers=N_LAYERS
        )
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        return self.fc_out(self.transformer(x)[:, CONTEXT_SIZE, :])


tokenizer = BrahmiTokenizer(VOCAB_PATH)
model_tr = CharTransformer(len(tokenizer.char2idx)).to(DEVICE)
model_tr.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location=DEVICE))
model_tr.eval()

# =========================
# UTILS
# =========================

def average_color(img):
    arr = np.array(img)
    return tuple(arr.mean(axis=(0, 1)).astype(int))


def mask_character(image_path, coords):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(coords, fill=average_color(img))
    img.save(MASKED_IMAGE)
    return MASKED_IMAGE


def decode_char_folder(name):
    return "".join(chr(int(p[2:], 16)) for p in name.split("_"))


# =========================
# PIPELINE
# =========================

def get_boxes(image_path):
    results = model_det(image_path, conf=0.25)
    detections = []

    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            detections.append({"bbox": box.tolist()})

    with open(DET_JSON, "w") as f:
        json.dump(detections, f)


@torch.no_grad()
def classify_detections(image_path, coords):
    image = Image.open(image_path).convert("RGB")

    with open(DET_JSON) as f:
        detections = json.load(f)

    results = []

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        crop = image.crop((x1, y1, x2, y2))
        x = transform(crop).unsqueeze(0).to(DEVICE)

        logits = model_clf(x)
        cls = classes[logits.argmax(1).item()]

        results.append({
            "char": decode_char_folder(cls),
            "bbox": d["bbox"]
        })

    results.append({"char": "_", "bbox": coords})

    with open(CLF_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def reconstruct_text():
    with open(CLF_JSON, encoding="utf-8") as f:
        data = json.load(f)

    data.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return "".join(d["char"] for d in data)


def predict_char(text):
    g = tokenizer.get_graphemes(text)
    idx = g.index("_")

    left = g[max(0, idx - CONTEXT_SIZE):idx]
    right = g[idx + 1:idx + 1 + CONTEXT_SIZE]

    ids = (
        [0] * (CONTEXT_SIZE - len(left)) +
        tokenizer.encode(left) +
        [1] +
        tokenizer.encode(right) +
        [0] * (CONTEXT_SIZE - len(right))
    )

    with torch.no_grad():
        pred = model_tr(torch.tensor([ids]).to(DEVICE)).argmax(1).item()

    return tokenizer.idx2char.get(pred, "")

# =========================
# API ROUTE
# =========================

@app.post("/ocr")
async def infer_missing_char(
    image: UploadFile = File(...),
    crop: str = Form(...)
):
    crop = json.loads(crop)

    with open(TMP_IMAGE, "wb") as f:
        shutil.copyfileobj(image.file, f)

    img = Image.open(TMP_IMAGE)
    iw, ih = img.size

    coords = [
        int(crop["x1"] * iw),
        int(crop["y1"] * ih),
        int(crop["x2"] * iw),
        int(crop["y2"] * ih),
    ]
    coords = [145,121,168,157] # missing char: 𑀴

    print("Crop coords (pixels):", coords)
    masked = mask_character(TMP_IMAGE, coords)
    get_boxes(masked)
    classify_detections(masked, coords)

    text = reconstruct_text()
    idx = text.index("_")

    before_char = text[:idx]
    after_char = text[idx + 1:]
    predicted_char = predict_char(text)


    print(before_char)
    print(predicted_char)
    print(after_char)
    return {
        "before_char": text[:idx],
        "predicted_char": predict_char(text),
        "after_char": text[idx + 1:]
    }
