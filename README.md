# 🏴‍☠️ One Piece Character Detector

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-green)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Detección automática de los 9 personajes principales de One Piece usando YOLOv8 entrenado con un dataset curado de Roboflow.**  
Proyecto académico para la asignatura de IA, donde se especializa un modelo de detección de objetos en el dominio del anime.

---

## Características

- ✅ Detecta los 9 personajes de la tripulación de los Sombrero de Paja (Luffy, Zoro, Sanji, Nami, Usopp, Chopper, Robin, Franky y Brook).
- ✅ Interfaz web local con **Gradio**.
- ✅ Dataset anotado a mano descargado desde **Roboflow**.
- ✅ Modelo **YOLOv8 Nano** — rápido y ligero, ejecutable en CPU.
- ✅ Compatible con Windows (probado en Python 3.12).

---

## Tecnologías Utilizadas

| Tecnología | Propósito |
|------------|-----------|
| [Python 3.12+](https://www.python.org/) | Lenguaje base |
| [Ultralytics YOLOv8](https://docs.ultralytics.com/) | Modelo de detección fine-tuneado |
| [Roboflow](https://roboflow.com/) | Dataset anotado `one-piece-uuyxt` |
| [Gradio](https://gradio.app/) | Interfaz de usuario interactiva |
| [OpenCV](https://opencv.org/) | Dibujo de bounding boxes |

---

## Estructura del Proyecto

```text
Analizador-Imagen/
│
├── src/
│   ├── app.py                  # Aplicación Gradio (interfaz web)
│   ├── train_yolo.py           # Fine-tuning de YOLOv8
│   └── descargar_roboflow.py   # Descarga el dataset desde Roboflow
│
├── ONE-PIECE-1/                # Dataset de Roboflow (excluido del repo)
│
├── runs/detect/                # Pesos del modelo entrenado (best.pt)
│
├── requirements.txt
└── README.md
```

---

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/Inaross/Analizador-Imagen.git
cd Analizador-Imagen
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. (Opcional) Descargar el dataset y reentrenar

Si quieres reentrenar el modelo desde cero:

```bash
python src/descargar_roboflow.py   # Descarga el dataset de Roboflow
python src/train_yolo.py           # Entrena YOLOv8 (50 épocas por defecto)
```

### 4. Lanzar la aplicación

```bash
python src/app.py
```

Abre `http://127.0.0.1:7860` en tu navegador.

---

## Uso de la Aplicación

1. **Sube una imagen** arrastrándola al panel de entrada.
2. **Ajusta el umbral de confianza** si lo deseas.
3. Haz clic en **"Detectar"** para ver los resultados con cajas delimitadoras y nombres.

---

## Resultados

El modelo YOLOv8 entrenado con el dataset de Roboflow alcanza alta precisión identificando los personajes incluso en imágenes de grupo, con tiempos de inferencia de ~10-30ms por imagen.

---

## Mejoras Futuras

- Ampliar el dataset con personajes secundarios y antagonistas.
- Desplegar la aplicación en Hugging Face Spaces para acceso público.

---

## Referencias

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Universe - One Piece Dataset](https://universe.roboflow.com/aivle5-f7j14/one-piece-uuyxt)
- [Gradio Documentation](https://gradio.app/docs/)
