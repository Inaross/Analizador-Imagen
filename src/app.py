"""
Interfaz Gradio para el detector de personajes de One Piece usando YOLOv8 entrenado.
"""

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Cargar modelo entrenado (ajusta la ruta)
model_path = "C:/Users/Usuario/Documents/Analizador-Imagen/Analizador-Imagen/runs/detect/models/yolo_one_piece/exp/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrena primero con train_yolo.py")

model = YOLO(model_path)

# Nombres de clases (debe coincidir con el entrenamiento)
class_names = ['Luffy', 'Zoro', 'Sanji', 'Nami', 'Usopp', 'Chopper', 'Robin']

def detect_objects(image, conf_threshold=0.25):
    # Ejecutar inferencia
    results = model(image, conf=conf_threshold)[0]
    
    # Dibujar resultados sobre la imagen
    img_array = np.array(image)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"
        # Dibujar rectángulo y texto
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Generar texto con detecciones
    detections_text = ""
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0].item()
        detections_text += f"{class_names[cls]}: {conf:.2f}\n"
    
    return Image.fromarray(img_array), detections_text

# Interfaz Gradio
with gr.Blocks(title="Detector de Personajes de One Piece") as demo:
    gr.Markdown("# 🏴‍☠️ Detector de Personajes de One Piece")
    gr.Markdown("Sube una imagen y el modelo detectará a Luffy, Zoro, Sanji, Nami, Usopp, Chopper y Robin.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Imagen")
            threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Umbral de confianza")
            btn = gr.Button("Detectar")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Resultado")
    
    text_output = gr.Textbox(label="Detecciones", lines=5)
    
    btn.click(fn=detect_objects, inputs=[image_input, threshold], outputs=[image_output, text_output])
    
    gr.Examples(
        examples=[["C:\\Users\\aleja\\Documents\\Analizador-Imagen\\data\\raw\\Luffy\\mw_file_0.png"], ["C:\\Users\\aleja\\Documents\\Analizador-Imagen\\data\\raw\\Luffy\\mw_file_1.png"]],
        inputs=image_input,
        outputs=[image_output, text_output],
        fn=detect_objects,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(share=False)