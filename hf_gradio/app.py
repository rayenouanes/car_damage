from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_2.pt"
CLASS_LABELS = {
    "crack": "Fissure (Crack)",
    "dent": "Bosse (Dent)",
    "glass shatter": "Vitre brisee (Glass shatter)",
    "lamp broken": "Feu/Phare casse (Lamp broken)",
    "scratch": "Rayure (Scratch)",
    "tire flat": "Pneu creve (Tire flat)",
}
COLORS = {
    "crack": (99, 208, 168),
    "dent": (249, 112, 102),
    "glass shatter": (91, 156, 246),
    "lamp broken": (251, 191, 36),
    "scratch": (192, 132, 252),
    "tire flat": (251, 146, 60),
}

MODEL = None


def get_model() -> YOLO:
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Le modele best_2.pt est introuvable dans le Space.")
        MODEL = YOLO(str(MODEL_PATH))
    return MODEL


def detect_damage(image: Image.Image, confidence: float, iou: float):
    if image is None:
        return None, []

    model = get_model()
    rgb = np.array(image.convert("RGB"))
    result = model.predict(
        source=rgb,
        conf=float(confidence),
        iou=float(iou),
        device="cpu",
        verbose=False,
    )[0]

    canvas = rgb.copy()
    rows = []
    names = {int(k): str(v) for k, v in model.names.items()}

    if result.boxes is not None:
        for idx, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            raw_name = names.get(cls_id, f"class_{cls_id}")
            label = CLASS_LABELS.get(raw_name, raw_name)
            color = COLORS.get(raw_name, (80, 200, 255))
            x1, y1, x2, y2 = xyxy.tolist()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            text = f"{label} {score:.2f}"
            cv2.putText(
                canvas,
                text,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            rows.append(
                {
                    "id": idx + 1,
                    "classe": label,
                    "confiance": round(score, 3),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )

    return Image.fromarray(canvas), rows


with gr.Blocks(title="Car Damage Detection") as demo:
    gr.Markdown("# Car Damage Detection")
    gr.Markdown("Uploade une image, puis lance la detection YOLO.")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Image voiture")
        image_output = gr.Image(type="pil", label="Resultat")
    with gr.Row():
        confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confiance")
        iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="IoU")
    run_button = gr.Button("Detecter", variant="primary")
    detections = gr.Dataframe(label="Detections", interactive=False)
    run_button.click(
        detect_damage,
        inputs=[image_input, confidence, iou],
        outputs=[image_output, detections],
    )


username = os.getenv("BACKOFFICE_USERNAME", "admin")
password = os.getenv("BACKOFFICE_PASSWORD", "")
auth = (username, password) if password else None

demo.launch(auth=auth)
