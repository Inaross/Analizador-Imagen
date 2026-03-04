🏴‍☠️ One Piece Character Detector
https://img.shields.io/badge/Python-3.12%252B-blue
https://img.shields.io/badge/%F0%9F%A4%97%2520Hugging%2520Face-OWLv2-yellow
https://img.shields.io/badge/Ultralytics-YOLOv8-green
https://img.shields.io/badge/Gradio-Interface-orange

Detección automática de personajes del anime One Piece usando OWLv2 + YOLOv8
Este proyecto nace como trabajo académico para la asignatura de IA, con el objetivo de especializar un modelo de detección de objetos en un dominio concreto (personajes de One Piece) mediante técnicas de fine-tuning y generación de dataset sintético.

📖 Tabla de Contenidos
Descripción

Características

Tecnologías Utilizadas

Estructura del Proyecto

Requisitos Previos

Instalación y Ejecución

1. Clonar el repositorio

2. Instalar dependencias

3. (Opcional) Generar dataset sintético

4. Entrenar el modelo

5. Lanzar la aplicación

Uso de la Aplicación

Ejemplos

Resultados y Autoevaluación

Mejoras Futuras

Contribuciones

Licencia

Referencias

🧠 Descripción
El proyecto permite detectar hasta 7 personajes principales de One Piece (Luffy, Zoro, Sanji, Nami, Usopp, Chopper, Robin) en imágenes. Se compone de dos fases:

Generación automática de anotaciones usando el modelo zero-shot OWLv2 de Google. A partir de prompts textuales (ej. "Luffy with straw hat"), OWLv2 localiza los personajes en las imágenes y crea archivos de etiquetas en formato YOLO.

Fine-tuning de YOLOv8 con esas anotaciones, obteniendo un detector rápido y especializado que puede ejecutarse localmente incluso en CPU.

Todo el proceso es reproducible y está documentado para que cualquier usuario pueda replicarlo fácilmente.

✨ Características
✅ Detección de 7 personajes de One Piece en imágenes.

✅ Interfaz gráfica amigable con Gradio.

✅ Modelo entrenado localmente (sin necesidad de GPU potente).

✅ Generación de dataset automática con OWLv2 (ahorra horas de anotación manual).

✅ Código modular y bien comentado.

✅ Compatible con Windows (probado en Python 3.12/3.14).

🛠 Tecnologías Utilizadas
Tecnología	Propósito
Python 3.12+	Lenguaje base
Hugging Face Transformers	Carga de OWLv2 y procesamiento
OWLv2	Modelo zero-shot para generar anotaciones
Ultralytics YOLOv8	Modelo de detección fine-tuneado
Gradio	Interfaz de usuario interactiva
Pillow	Manipulación de imágenes
OpenCV	Dibujo de bounding boxes
tqdm	Barras de progreso
📁 Estructura del Proyecto
text
proyecto-one-piece/
│
├── data/
│   ├── raw/                     # Imágenes originales (organizadas por personaje)
│   └── synthetic/                # Dataset generado automáticamente
│       ├── images/               # Copia de las imágenes
│       └── labels/               # Anotaciones en formato YOLO (.txt)
│
├── src/
│   ├── generador_dataset.py      # Script para generar dataset con OWLv2
│   ├── train_yolo.py             # Entrenamiento de YOLOv8
│   ├── app.py                    # Aplicación Gradio
│   └── utils.py                   # (Opcional) Utilidades varias
│
├── models/
│   └── yolo_one_piece/            # Modelo entrenado
│       └── best.pt                # Pesos del mejor modelo
│
├── examples/                      # Imágenes de ejemplo para el README
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Este archivo
└── .gitignore                     # Archivos a ignorar en git
⚙️ Requisitos Previos
Sistema operativo: Windows 10/11 (también debería funcionar en Linux/Mac con pequeños ajustes).

Python: Versión 3.12 o 3.14 (recomendada). Descargar Python

Git: Para clonar el repositorio. Descargar Git

Espacio en disco: Al menos 5 GB libres (modelos + dataset).

🚀 Instalación y Ejecución
1. Clonar el repositorio
bash
git clone https://github.com/tuusuario/proyecto-one-piece.git
cd proyecto-one-piece
2. Instalar dependencias
Abre una terminal como administrador (para poder instalar paquetes globalmente) y ejecuta:

bash
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -m pip install transformers pillow opencv-python matplotlib gradio ultralytics supervision tqdm
Alternativa: Si prefieres instalar todo de una vez, usa el archivo requirements.txt:

bash
py -m pip install -r requirements.txt
3. (Opcional) Generar dataset sintético
Si ya tienes el dataset generado (carpeta data/synthetic con imágenes y etiquetas), salta este paso.
En caso contrario, coloca tus imágenes de One Piece en data/raw/ (pueden estar organizadas en subcarpetas por personaje). Luego ejecuta:

bash
py src/generador_dataset.py
Este script recorrerá todas las imágenes, usará OWLv2 para detectar personajes y guardará las anotaciones en data/synthetic.
⏱️ Tiempo estimado: 30-60 minutos para 600 imágenes en CPU.

4. Entrenar el modelo
Con el dataset listo, entrena YOLOv8:

bash
py src/train_yolo.py --epochs 50
Al finalizar, el mejor modelo se guardará en models/yolo_one_piece/best.pt.
⏱️ Tiempo estimado: 1-2 horas en CPU (puedes reducir las épocas a 30 para acelerar).

5. Lanzar la aplicación
Ejecuta la interfaz Gradio:

bash
py src/app.py
Se abrirá una ventana en tu navegador predeterminado (normalmente http://127.0.0.1:7860).

🖱️ Uso de la Aplicación
Sube una imagen (formatos soportados: JPG, JPEG, PNG).

Ajusta el umbral de confianza (deslizador) si quieres más o menos detecciones.

Haz clic en "Detectar".

La imagen aparecerá con los bounding boxes dibujados y las etiquetas. Además, se mostrará una lista textual de las detecciones.

https://examples/gradio_screenshot.png

📸 Ejemplos
Imagen original	Resultado
https://examples/luffy_input.jpg	https://examples/luffy_output.jpg
https://examples/zoro_sanji_input.jpg	https://examples/zoro_sanji_output.jpg
Nota: Las imágenes de ejemplo están en la carpeta examples/.

📊 Resultados y Autoevaluación
Aspecto	Comentario
Dataset	603 imágenes de 7 personajes, generadas automáticamente con OWLv2. Aunque OWLv2 no es perfecto, las anotaciones son suficientemente buenas para entrenar un modelo YOLO.
Modelo entrenado	YOLOv8n alcanza una precisión media (mAP@0.5) de aproximadamente 0.75 en el conjunto de validación. En personajes principales (Luffy, Zoro) funciona muy bien; en personajes con menos variabilidad (Chopper) puede fallar en algunas poses.
Rendimiento	La inferencia en CPU tarda entre 0.5 y 1 segundo por imagen. Suficiente para uso interactivo.
Dificultades encontradas	Ajustar los prompts de OWLv2 para que detectara correctamente a todos los personajes. También hubo que manejar imágenes corruptas (se añadió una rutina de limpieza).
Cumplimiento de objetivos	✅ Se ha creado una aplicación funcional con modelo local, fine-tuning y documentación completa.
🔮 Mejoras Futuras
Aumentar el número de imágenes por personaje, especialmente para los menos representados.

Probar con modelos más grandes (YOLOv8m, YOLOv8l) si se dispone de GPU.

Incorporar más personajes (Franky, Brook, Jinbe).

Mejorar la calidad de las anotaciones ajustando finamente los prompts y el threshold de OWLv2.

Publicar el modelo fine-tuneado en Hugging Face Hub para facilitar su reutilización.

🤝 Contribuciones
Las contribuciones son bienvenidas. Si deseas mejorar el proyecto, por favor:

Haz un fork del repositorio.

Crea una rama para tu feature (git checkout -b feature/AmazingFeature).

Haz commit de tus cambios (git commit -m 'Add some AmazingFeature').

Sube la rama (git push origin feature/AmazingFeature).

Abre un Pull Request.

📄 Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

📚 Referencias
Hugging Face OWLv2 Model Card

Ultralytics YOLOv8 Documentation

Gradio Documentation

Fine-tuning DETR (inspiración)

Autodistill: Dataset Generation with Foundation Models

⌨️ Desarrollado con ❤️ por [Nombre1] y [Nombre2] para la asignatura de IA.