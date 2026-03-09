# 🏴‍☠️ One Piece Character Detector

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-green)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange)

### Título
One Piece Character Detector - Detección de objetos en Anime (Visión Artificial)

### Integrantes
Alejandro Bueno Ortiz, Alexander Gavilanez Castro

---

## Descripción
**¿Qué hace nuestra app?**  
Esta aplicación es un detector de personajes de *One Piece* capaz de identificar automáticamente a los 9 miembros principales de la tripulación de los Sombrero de Paja (Luffy, Zoro, Sanji, Nami, Usopp, Chopper, Robin, Franky y Brook) en imágenes. Cuenta con una interfaz web local amigable construida en Gradio.

**¿Qué problema resuelve?**  
Resuelve el reto de identificar personajes específicos dentro del característico estilo de dibujo del anime, el cual presenta desafíos únicos para los modelos tradicionales de visión artificial debido a sus trazos, colores atípicos y proporciones exageradas. Sirve como herramienta de etiquetado automático para fans, categorización de imágenes o creadores de contenido.

---

## Modelo Base
**¿Qué modelo de Hugging Face elegisteis y por qué?**  
Elegimos **YOLOv8 Nano** (`yolov8n.pt`) de Ultralytics (frecuentemente exportado y referenciado en el ecosistema de Hugging Face para visión). Elegimos YOLOv8 por su asombrosa velocidad de inferencia a tiempo real y su equilibrio perfecto entre precisión y rendimiento. La versión "Nano" es lo suficientemente ligera como para ser entrenada rápidamente y ejecutada en CPU, lo cual es ideal para una aplicación web interactiva.

---

## Técnica de Adaptación: Fine-tuning
Para este proyecto utilizamos **Fine-tuning** (Ajuste Fino) en lugar de RAG (Retrieval-Augmented Generation).

**¿Por qué elegimos Fine-tuning sobre RAG?**  
RAG es una técnica diseñada para modelos de lenguaje (LLMs) donde se inyecta texto como contexto para responder preguntas. Nuestro problema es puramente visual (Detección de Objetos). Para que un modelo de visión aprenda a detectar clases nuevas que no venían en su entrenamiento original, es necesario enseñarle a reconocer esas formas y patrones visuales modificando sus pesos internos, lo cual solo es posible con Fine-tuning.

**¿Cómo funciona?**  
Tomamos el modelo YOLOv8 preentrenado (que ya sabe detectar formas básicas y objetos cotidianos gracias al dataset COCO) y lo reentrenamos (*transfer learning*) con nuestro propio conjunto de imágenes de *One Piece*. El modelo retiene su capacidad de extraer características visuales, pero ajusta sus últimas capas para predecir las *bounding boxes* (cajas delimitadoras) y las probabilidades de nuestras 9 clases nuevas.

---

## Dataset
**¿De dónde sacasteis los datos?**  
El dataset fue obtenido de [Roboflow Universe](https://universe.roboflow.com/aivle5-f7j14/one-piece-uuyxt), una plataforma de visión artificial comunitaria. Utilizamos una versión específicamente anotada para YOLOv8.

**¿Cómo los procesasteis?**  
1. Creamos el script `descargar_roboflow.py` que se conecta a la API de Roboflow usando nuestra clave.
2. Descargamos el dataset estructurado en formato YOLO (imágenes y etiquetas de coordenadas divididas en `train`, `valid` y `test` con su archivo `data.yaml`).
3. El dataset ya traía las *bounding boxes* aplicadas y balanceadas para los 9 personajes.
4. Pasamos la ruta absoluta de este dataset directamente al motor de entrenamiento de Ultralytics para el fine-tuning.

---

## Instrucciones de Instalación y Ejecución

Sigue estos pasos desde clonar el repositorio hasta hacer la primera inferencia:

### 1. Clonar el repositorio
Abre una terminal en tu ordenador y ejecuta:
```bash
git clone https://github.com/tu-usuario/Analizador-Imagen.git
cd Analizador-Imagen
```

### 2. Instalar dependencias
Asegúrate de tener **Python 3.12+**. Instala las librerías necesarias:
```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset desde Roboflow
Ejecuta nuestro script para descargar las imágenes y etiquetas. *Nota: esto creará una carpeta llamada `ONE-PIECE-1` en el directorio raíz*.
```bash
python src/descargar_roboflow.py
```

### 4. Entrenar el modelo (Fine-tuning)
Ejecuta el script de entrenamiento para adaptar YOLOv8 a los personajes. (Generará el modelo final en `models/yolo_one_piece/best.pt`). Por defecto entrena por 50 épocas.
```bash
python src/train_yolo.py
```

### 5. Lanzar la aplicación (Hacer la primera "pregunta"\*)
Levanta la interfaz gráfica local basada en Gradio:
```bash
python src/app.py
```
*\*En nuestro caso, la "pregunta" es subir una imagen:*
1. Abre tu navegador web en la dirección `http://127.0.0.1:7860` (te lo indicará la terminal).
2. Sube una foto en la que aparezcan personajes de One Piece en el recuadro.
3. Haz clic en **"Detectar"**.
4. ¡El modelo dibujará las cajas sobre los personajes detectados!

---
*(Estructura del Proyecto)*
```text
Analizador-Imagen/
├── src/
│   ├── app.py                  # Aplicación Gradio (interfaz web)
│   ├── train_yolo.py           # Fine-tuning de YOLOv8
│   └── descargar_roboflow.py   # Descarga el dataset desde Roboflow
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este documento
```
