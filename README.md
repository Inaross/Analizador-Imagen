# 🏴‍☠️ One Piece Character Detector

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-OWLv2-yellow)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-green)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Detección automática de personajes del anime One Piece usando OWLv2 + YOLOv8**  
Proyecto académico para la asignatura de IA, donde se especializa un modelo de detección de objetos en un dominio concreto mediante generación de dataset sintético y fine-tuning.

---

## 📖 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación y Ejecución](#-instalación-y-ejecución)
  - [1. Clonar el repositorio](#1-clonar-el-repositorio)
  - [2. Instalar dependencias](#2-instalar-dependencias)
  - [3. (Opcional) Generar dataset sintético](#3-opcional-generar-dataset-sintético)
  - [4. Entrenar el modelo](#4-entrenar-el-modelo)
  - [5. Lanzar la aplicación](#5-lanzar-la-aplicación)
- [Uso de la Aplicación](#-uso-de-la-aplicación)
- [Ejemplos](#-ejemplos)
- [Resultados y Autoevaluación](#-resultados-y-autoevaluación)
- [Mejoras Futuras](#-mejoras-futuras)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)
- [Referencias](#-referencias)

---

## 🧠 Descripción

Este proyecto permite **detectar hasta 7 personajes principales** de One Piece (Luffy, Zoro, Sanji, Nami, Usopp, Chopper, Robin) en imágenes. Se compone de dos fases:

1. **Generación automática de anotaciones** usando el modelo zero-shot **OWLv2** de Google. A partir de prompts textuales (ej. *"Luffy with straw hat"*), OWLv2 localiza los personajes en las imágenes y crea archivos de etiquetas en formato YOLO.
2. **Fine-tuning de YOLOv8** con esas anotaciones, obteniendo un detector rápido y especializado que puede ejecutarse localmente incluso en CPU.

Todo el proceso es **reproducible** y está documentado para que cualquier usuario pueda replicarlo fácilmente.

---

## ✨ Características

- ✅ Detección de 7 personajes de One Piece en imágenes.
- ✅ Interfaz gráfica amigable con **Gradio**.
- ✅ Modelo entrenado localmente (sin necesidad de GPU potente).
- ✅ Generación de dataset automática con OWLv2 (ahorra horas de anotación manual).
- ✅ Código modular y bien comentado.
- ✅ Compatible con Windows (probado en Python 3.12/3.14).

---

## 🛠 Tecnologías Utilizadas

| Tecnología | Propósito |
|------------|-----------|
| [Python 3.12+](https://www.python.org/) | Lenguaje base |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) | Carga de OWLv2 y procesamiento |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Modelo zero-shot para generar anotaciones |
| [Ultralytics YOLOv8](https://docs.ultralytics.com/) | Modelo de detección fine-tuneado |
| [Gradio](https://gradio.app/) | Interfaz de usuario interactiva |
| [Pillow](https://python-pillow.org/) | Manipulación de imágenes |
| [OpenCV](https://opencv.org/) | Dibujo de bounding boxes |
| [tqdm](https://tqdm.github.io/) | Barras de progreso |

---

## 📁 Estructura del Proyecto
Proyecto-one-piece/
│
├── data/
│ ├── raw/ # Imágenes originales (organizadas por personaje)
│ └── synthetic/ # Dataset generado automáticamente
│ ├── images/ # Copia de las imágenes
│ └── labels/ # Anotaciones en formato YOLO (.txt)
│
├── src/
│ ├── generador_dataset.py # Script para generar dataset con OWLv2
│ ├── train_yolo.py # Entrenamiento de YOLOv8
│ ├── app.py # Aplicación Gradio
│ └── utils.py # (Opcional) Utilidades varias
│
├── models/
│ └── yolo_one_piece/ # Modelo entrenado
│ └── best.pt # Pesos del mejor modelo
│
├── examples/ # Imágenes de ejemplo para el README
├── requirements.txt # Dependencias del proyecto
├── README.md # Este archivo
└── .gitignore # Archivos a ignorar en git

---

## ⚙️ Requisitos Previos

* **Sistema operativo:** Windows 10/11 (también debería funcionar en Linux/Mac con pequeños ajustes).
* **Python:** Versión 3.12 o 3.14 (recomendada). [Descargar Python](https://www.python.org/downloads/)
* **Git:** Para clonar el repositorio. [Descargar Git](https://git-scm.com/downloads)
* **Espacio en disco:** Al menos 5 GB libres (modelos + dataset).

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/Inaross/Analizador-Imagen.git
cd Analizador-Imagen
```

### 2. Instalar dependencias
Instalar los paquetes requeridos
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Generar dataset sintético
 Si deseas generar las anotaciones desde las imágenes en `data/raw` utilizando OWLv2:

```bash
python src/generador_dataset.py
```

*Nota: Este proceso puede tardar dependiendo de tu procesador y cantidad de imágenes.*

### 4. Entrenar el modelo

Para realizar el fine-tuning de YOLOv8 con el dataset sintético recién creado:

```bash
python src/train_yolo.py
```

Al finalizar, el modelo se guardará automáticamente en `models/yolo_one_piece/best.pt`.

### 5. Lanzar la aplicación

Una vez entrenado (o si ya dispones del archivo `best.pt`), arranca la interfaz web:

```bash
python src/app.py
```

## 🖥️ Uso de la Aplicación

1. **Abre la interfaz** web en tu navegador.
2. **Sube una imagen** arrastrándola al panel de entrada o haciendo clic para buscar en tu equipo.
3. **Ajusta el umbral de confianza** *(Confidence Threshold)* si lo deseas mediante el deslizador para ser más o menos estricto con las detecciones.
4. Haz clic en **"Detectar"** y visualiza los resultados en el panel derecho con sus respectivas cajas delimitadoras y etiquetas.

## 📊 Resultados y Autoevaluación

* **Precisión Zero-Shot (OWLv2):** Logra un gran rendimiento inicial con descripciones detalladas, facilitando la creación del *ground truth*.
* **Rendimiento YOLOv8:** El fine-tuning permite pasar de tiempos de inferencia altos (OWLv2) a detecciones en tiempo real (~20-50ms por imagen) manteniendo una alta precisión en las clases objetivo.

## 🚀 Mejoras Futuras

* **Ampliar el Dataset:** Incluir personajes secundarios y antagonistas.
* **Aumento de Datos (Data Augmentation):** Aplicar rotaciones, cambios de brillo y ruido para hacer el modelo más robusto.
* **Despliegue en la Nube:** Alojar la aplicación en plataformas como Hugging Face Spaces para acceso público permanente.

## 🤝 Contribuciones

Las contribuciones son bienvenidas para mejorar este proyecto. Los pasos son sencillos:

1. Haz un **Fork** del repositorio.
2. Crea tu rama para la nueva característica (`git checkout -b feature/NuevaCaracteristica`).
3. Haz **Commit** de tus cambios (`git commit -m "Añadir nueva característica"`).
4. Haz **Push** a la rama (`git push origin feature/NuevaCaracteristica`).
5. Abre un **Pull Request**.

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Para más detalles, consulta el archivo `LICENSE` incluido en el repositorio.

## 📚 Referencias

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* [OWLv2: Scaling Open-Vocabulary Object Detection](https://huggingface.co/docs/transformers/model_doc/owlv2)
* [Gradio Documentation](https://gradio.app/docs/)
