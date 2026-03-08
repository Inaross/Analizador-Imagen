"""
Entrena YOLOv8 con el dataset sintético.
"""

import os
import yaml
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/synthetic")
    parser.add_argument("--output_dir", type=str, default="./models/yolo_one_piece")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # Crear archivo data.yaml para YOLO
    data_yaml = {
        'path': os.path.abspath(args.data_dir),
        'train': 'images/train',  
        'val': 'images/val',
        'nc': 9,  # Actualizado a 9 clases
        'names': ['Luffy', 'Zoro', 'Sanji', 'Nami', 'Usopp', 'Chopper', 'Robin', 'Franky', 'Brook']  # Orden exacto del generador
    }
    yaml_path = os.path.join(args.data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

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