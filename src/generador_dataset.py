import os
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection 
from tqdm import tqdm
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "Luffy face character straw hat",
        "Zoro face character green hair",
        "Sanji face character blonde hair",
        "Nami face character orange hair",
        "Usopp face character long nose",
        "Chopper face character small reindeer",
        "Robin face character black hair"
    ])
    args = parser.parse_args()

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Crear carpetas de salida
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)

    # Obtener todas las imágenes recursivamente
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))

    print(f"Se encontraron {len(image_files)} imágenes en total.")

    errores = 0
    for img_path in tqdm(image_files, desc="Procesando imágenes"):
        try:
            # Nombre base sin extensión
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Abrir imagen (intentamos con manejo de truncados)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"\nError al abrir {img_path}: {e}. Se omite.")
                errores += 1
                continue

            width, height = image.size

            # Inferencia
            inputs = processor(text=args.prompts, images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[height, width]])
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=args.threshold
            )[0]

            # Guardar etiquetas en formato YOLO
            label_file = os.path.join(args.output_dir, "labels", base_name + ".txt")
            with open(label_file, 'w') as f:
                for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
                    class_id = label_id.item()
                    x1, y1, x2, y2 = box.tolist()
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            # Copiar imagen a la carpeta de salida
            image.save(os.path.join(args.output_dir, "images", os.path.basename(img_path)))

        except Exception as e:
            print(f"\nError inesperado procesando {img_path}: {e}")
            errores += 1
            continue

    print(f"Procesamiento completado. {len(image_files) - errores} imágenes procesadas correctamente, {errores} errores.")
    print("Dataset sintético generado en:", args.output_dir)

if __name__ == "__main__":
    main()