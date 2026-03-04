"""
Genera un dataset sintético usando OWLv2 para etiquetar imágenes de One Piece.
Las anotaciones se guardan en formato YOLO (txt por imagen).
"""

import os
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "Luffy with straw hat",
        "Zoro with green hair and swords",
        "Sanji with blonde hair and curly eyebrow",
        "Nami with orange hair",
        "Usopp with long nose"
    ])
    args = parser.parse_args()

    # Cargar modelo OWLv2
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Crear carpetas de salida
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)

    # Mapeo de prompts a IDs de clase (para YOLO)
    class_names = [p.split()[0] for p in args.prompts]  # Toma la primera palabra como clase
    class_to_id = {name: i for i, name in enumerate(class_names)}

    # Procesar cada imagen
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(image_files, desc="Procesando imágenes"):
        img_path = os.path.join(args.input_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Inferencia con OWLv2
        inputs = processor(text=args.prompts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[height, width]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=args.threshold
        )[0]

        # Crear archivo de etiquetas en formato YOLO
        label_file = os.path.join(args.output_dir, "labels", img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_file, 'w') as f:
            for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
                # label_id corresponde al índice del prompt
                class_id = label_id.item()
                # Convertir box de [x1,y1,x2,y2] a [x_center, y_center, width, height] normalizado
                x1, y1, x2, y2 = box.tolist()
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # Copiar imagen a la carpeta de salida
        image.save(os.path.join(args.output_dir, "images", img_file))

    print("Dataset sintético generado en:", args.output_dir)

if __name__ == "__main__":
    main()