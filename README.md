# Detector de Personajes de One Piece

**Integrantes:** [Nombre1] y [Nombre2]

## Descripción
Aplicación de detección de objetos que identifica personajes del anime One Piece (Luffy, Zoro, Sanji, Nami, Usopp) en imágenes. Utiliza un modelo YOLOv8 fine-tuneado con un dataset generado automáticamente por OWLv2 (zero-shot). Todo el proceso es local y reproducible.

## Modelo Base
- **Generador de dataset:** [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) – modelo zero-shot que utilizamos para etiquetar nuestras imágenes sin necesidad de anotación manual.
- **Modelo final:** YOLOv8n (nano) – rápido y ligero, ideal para ejecución local.

## Técnica de Adaptación: Fine-tuning con Dataset Sintético
Elegimos este enfoque porque:
- OWLv2 nos permite crear rápidamente un dataset personalizado con prompts en lenguaje natural.
- Luego fine-tuneamos YOLOv8 con esas anotaciones, obteniendo un modelo especializado y eficiente.
- Es una combinación de zero-shot y fine-tuning que demuestra comprensión de ambas técnicas.

## Dataset
- **Fuente:** Imágenes de One Piece recolectadas manualmente (carpeta `data/raw/`).
- **Generación:** OWLv2 etiqueta automáticamente las imágenes usando prompts como "Luffy with straw hat", "Zoro with green hair", etc. (script `generate_dataset.py`).
- **Formato final:** YOLO (archivos .txt por imagen) en `data/synthetic/`.

## Instrucciones de instalación y ejecución

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/proyecto-one-piece-detection.git
   cd proyecto-one-piece-detection