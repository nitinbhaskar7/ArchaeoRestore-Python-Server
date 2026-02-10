import io
import json
import torch
import torch.nn as nn
import numpy as np
import regex as re
import shutil
import os

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

YOLO_MODEL_PATH = "models/yolo_det_new.pt"
CLF_MODEL_PATH = "models/mobilenet_brahmi_classifier_new.pt"
TRANSFORMER_MODEL_PATH = "models/transformer.pth"
VOCAB_PATH = "vocab.json"

TMP_IMAGE_PATH = "outputs/input.jpg"
MASKED_IMAGE_PATH = "outputs/masked.jpg"
OUTPUT_DET_JSON = "outputs/detection_results.json"
OUTPUT_CLF_JSON = "outputs/classification_results.json"

CONTEXT_SIZE = 5
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
WINDOW_SIZE = (CONTEXT_SIZE * 2) + 1

os.makedirs("outputs", exist_ok=True)

# =========================
# FASTAPI INIT
# =========================

app = FastAPI(title="Brahmi OCR Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODELS (ONCE)
# =========================

model_det = YOLO(YOLO_MODEL_PATH)

ckpt = torch.load(CLF_MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]

model_clf = mobilenet_v2(pretrained=False)
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
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.unk_idx = self.char2idx.get("<unk>", 2)

    def get_graphemes(self, text):
        return re.findall(r"\X", text)

    def encode(self, graphemes):
        return [self.char2idx.get(g, self.unk_idx) for g in graphemes]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, WINDOW_SIZE, EMBED_DIM)
        )
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
        x = self.transformer(x)
        return self.fc_out(x[:, CONTEXT_SIZE, :])



tokenizer = BrahmiTokenizer(VOCAB_PATH)
model_tr = CharTransformer(len(tokenizer.char2idx)).to(DEVICE)
model_tr.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE))
model_tr.eval()

# =========================
# UTILS
# =========================

def average_color(image):
    arr = np.array(image)
    return tuple(arr.mean(axis=(0, 1)).astype(int))


def mask_character(image_path, coords_xy):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(coords_xy, fill=average_color(image))
    image.save(MASKED_IMAGE_PATH)
    return MASKED_IMAGE_PATH


def decode_char_folder(name: str) -> str:
    chars = []
    for part in name.split("_"):
        chars.append(chr(int(part[2:], 16)))
    return "".join(chars)

# =========================
# PIPELINE
# =========================

def get_boxes(image_path):
    results = model_det(image_path, conf=0.25)
    detections = []

    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            detections.append({"bbox": box.tolist()})

    with open(OUTPUT_DET_JSON, "w") as f:
        json.dump(detections, f)


@torch.no_grad()
def classify_detections(image_path, coords_xy):
    image = Image.open(image_path).convert("RGB")

    with open(OUTPUT_DET_JSON, "r") as f:
        detections = json.load(f)

    results = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        crop = image.crop((x1, y1, x2, y2))
        x = transform(crop).unsqueeze(0).to(DEVICE)

        logits = model_clf(x)
        pred = classes[logits.argmax(1).item()]

        results.append({
            "char": decode_char_folder(pred),
            "bbox": det["bbox"]
        })

    results.append({"char": "_", "bbox": coords_xy})

    with open(OUTPUT_CLF_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def reconstruct_text():
    with open(OUTPUT_CLF_JSON, "r", encoding="utf-8") as f:
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
        logits = model_tr(torch.tensor([ids]).to(DEVICE))
        pred = logits.argmax(1).item()

    return tokenizer.idx2char.get(pred, "")

# =========================
# API ROUTE
# =========================

@app.post("/ocr")
async def infer(
    image: UploadFile = File(...),
    crop: str = Form(...)
):

    # Open image with Pillow
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))
    # close the upload file

    temp = json.loads(crop)
    
    print("Crop coords:", temp)
    # coords_xy = [int(i) for i in temp]  # expects [x1,y1,x2,y2]
    # coords_xy = json.loads(coords_xy)  # expects [x1,y1,x2,y2]
    crop_data = json.loads(crop)
    print("Crop data:", crop_data)

            # Draw rectangle if crop exists
    iw, ih = img.size
    x1 = int(crop_data['x1'] * iw)
    y1 = int(crop_data['y1'] * ih)
    x2 = int(crop_data['x2'] * iw)
    y2 = int(crop_data['y2'] * ih)
    coords_xy = [x1, y1, x2, y2]
    print("Computed coords:", coords_xy)
    with open(TMP_IMAGE_PATH, "wb") as f:
        f.write(img_bytes)

    masked = mask_character(TMP_IMAGE_PATH, coords_xy)
    get_boxes(masked)
    classify_detections(masked, coords_xy)

    text = reconstruct_text()
    idx = text.index("_")
    return {
        "before_char": text[:idx],
        "predicted_char": predict_char(text),
        "after_char": text[idx + 1:]
    }

@app.post("/test")
async def receive_image(
    image: UploadFile = File(...),
    crop: str = Form(None)
):
    print(f"Received image: {image.filename}")

    # Open image with Pillow
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Parse crop data
    crop_data = None
    if crop:
        try:
            crop_data = json.loads(crop)
            print("Crop data:", crop_data)

            # Draw rectangle if crop exists
            iw, ih = img.size
            x1 = int(crop_data['x1'] * iw)
            y1 = int(crop_data['y1'] * ih)
            x2 = int(crop_data['x2'] * iw)
            y2 = int(crop_data['y2'] * ih)
            print("Computed coords:", [x1, y1, x2, y2]) 
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        except Exception as e:
            print("Invalid crop data:", e)

    # Save the image with rectangle
   

    return {
        "status": "success",
        "message": "Image received and rectangle drawn successfully",
        "crop": crop_data,
        "ocrText": "Sample OCR text"
    }
