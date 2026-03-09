"""
Entrena YOLOv8 con el dataset sintético.
"""

import os
import yaml
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./models/yolo_one_piece")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # Inicializar modelo de YOLOv8 Nano para entrenamiento rápido
    model = YOLO('yolov8n.pt')

    # Path absoluto al data.yaml descargado desde Roboflow (ONE-PIECE-1)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml_path = os.path.join(base_dir, 'ONE-PIECE-1', 'data.yaml')

    # Entrenar
    model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=25,  # Añadimos paciencia para evitar overfitting prolongado
        project=args.output_dir,
        name='exp',
        exist_ok=True
    )

    # Guardar el mejor modelo
    best_model_path = os.path.join(args.output_dir, 'exp', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        os.rename(best_model_path, os.path.join(args.output_dir, 'best.pt'))
    print("Entrenamiento completado. Modelo guardado en:", args.output_dir)

if __name__ == "__main__":
    main()