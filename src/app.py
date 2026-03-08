"""
Interfaz Gradio para el detector de personajes de One Piece usando YOLOv8 entrenado.
"""

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Buscamos el mejor modelo de forma robusta con rutas absolutas
base_dir = Path(__file__).parent.parent
model_path = None
for p in base_dir.rglob('best.pt'):
    if 'yolo_one_piece' in p.parts:
        model_path = str(p)
        break

if not model_path or not os.path.exists(model_path):
    raise FileNotFoundError("No se encontró el modelo best.pt. Entrena primero con train_yolo.py")

model = YOLO(model_path)

# Corrección de nombres del dataset de Roboflow
NOMBRE_CORREGIDO = {
    'Blook': 'Brook',
    'Chyopa': 'Chopper',
    'Sangdi': 'Sanji',
    'NicoRobin': 'Robin',
}

def detect_objects(image, conf_threshold=0.25):
    results = model(image, conf=conf_threshold)[0]
    img_array = np.array(image.convert("RGB"))

    detections_text = ""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0])
        label_name = model.names[cls]
        label_name = NOMBRE_CORREGIDO.get(label_name, label_name)

        label = f"{label_name} {conf:.2f}"
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detections_text += f"{label_name}: {conf:.2f}\n"

    if not detections_text:
        detections_text = "No se detectaron personajes."

    return Image.fromarray(img_array), detections_text

# Interfaz Gradio
with gr.Blocks(title="Detector de Personajes de One Piece") as demo:
    gr.Markdown("# 🏴‍☠️ Detector de la Tripulación de los Sombrero de Paja")
    gr.Markdown("Sube una imagen y el modelo **YOLOv8** detectará a Luffy, Zoro, Sanji, Nami, Usopp, Chopper, Robin, Franky o Brook.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Imagen")
            threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Umbral de confianza")
            btn = gr.Button("Detectar", variant="primary")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Resultado")

    text_output = gr.Textbox(label="Detecciones", lines=5)
    btn.click(fn=detect_objects, inputs=[image_input, threshold], outputs=[image_output, text_output])

if __name__ == "__main__":
    demo.launch(share=False)