"""
PROYECTO: Detector de objetos Zero-Shot con Hugging Face
Autor: [Tu nombre]
Fecha: [Fecha]
Descripción: Este script utiliza el modelo OWLv2 de Google para detectar objetos
en imágenes a partir de una lista de etiquetas textuales. Incluye:
- Carga del modelo mediante pipeline.
- Función para dibujar bounding boxes sobre la imagen.
- Procesamiento de una imagen local.
- Guardado del resultado.
- (Opcional) Interfaz gráfica con Gradio.
"""

# -------------------------------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS
# -------------------------------------------------------------------
import os
import sys
from transformers import pipeline          # Pipeline de Hugging Face
from PIL import Image, ImageDraw, ImageFont  # Manipulación de imágenes
import argparse                            # Para argumentos de línea de comandos (opcional)

# -------------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# -------------------------------------------------------------------
# Nombre del modelo en Hugging Face Hub
MODELO = "google/owlv2-base-patch16-ensemble"

# -------------------------------------------------------------------
# 1. CARGA DEL MODELO (pipeline)
# -------------------------------------------------------------------
print("Cargando modelo... (esto puede tomar unos segundos la primera vez)")
try:
    # El pipeline se encarga de descargar y preparar el modelo automáticamente
    detector = pipeline(
        model=MODELO,
        task="zero-shot-object-detection"
    )
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# 2. FUNCIÓN PARA DIBUJAR LAS DETECCIONES EN LA IMAGEN
# -------------------------------------------------------------------
def dibujar_detecciones(imagen_original, predicciones):
    """
    Dibuja rectángulos y etiquetas sobre la imagen para cada objeto detectado.

    Parámetros:
        imagen_original (PIL.Image): Imagen sin modificar.
        predicciones (list): Lista de diccionarios con 'box', 'label', 'score'.

    Retorna:
        PIL.Image: Imagen con las anotaciones dibujadas.
    """
    # Crear una copia para no alterar la original
    img = imagen_original.copy()
    draw = ImageDraw.Draw(img)

    # Intentar cargar una fuente para el texto (si no, usa la predeterminada)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except:
        font = ImageFont.load_default()

    # Recorrer todas las predicciones
    for pred in predicciones:
        # Extraer coordenadas del bounding box
        box = pred['box']   # Formato: {'xmin': ..., 'ymin': ..., 'xmax': ..., 'ymax': ...}
        label = pred['label']
        score = pred['score']

        # Dibujar rectángulo rojo de grosor 3
        draw.rectangle(
            [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
            outline="red",
            width=3
        )

        # Texto a mostrar: "etiqueta confianza"
        texto = f"{label} {score:.2f}"

        # Calcular el rectángulo de fondo del texto
        bbox = draw.textbbox((box['xmin'], box['ymin']), texto, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((box['xmin'], box['ymin']), texto, fill="white", font=font)

    return img

# -------------------------------------------------------------------
# 3. FUNCIÓN PRINCIPAL DE DETECCIÓN
# -------------------------------------------------------------------
def detectar_objetos(ruta_imagen, candidatos, umbral=0.1):
    """
    Procesa una imagen y devuelve la imagen anotada y un resumen de texto.

    Parámetros:
        ruta_imagen (str): Ruta al archivo de imagen.
        candidatos (list): Lista de strings con los objetos a buscar.
        umbral (float): Confianza mínima para considerar una detección.

    Retorna:
        tuple: (imagen_con_cajas, texto_resultado)
    """
    # Verificar que la imagen existe
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # Abrir imagen con PIL
    imagen = Image.open(ruta_imagen)
    print(f"Imagen cargada: {ruta_imagen} (tamaño: {imagen.size})")

    # Ejecutar la detección
    print("Detectando objetos...")
    predicciones = detector(imagen, candidate_labels=candidatos, threshold=umbral)
    print(f"Se encontraron {len(predicciones)} detecciones.")

    # Dibujar las cajas sobre la imagen
    imagen_con_cajas = dibujar_detecciones(imagen, predicciones)

    # Crear resumen de texto
    texto = ""
    for i, p in enumerate(predicciones, 1):
        texto += f"{i}. {p['label']} (confianza: {p['score']:.2f}) - coordenadas: {p['box']}\n"
    if not texto:
        texto = "No se detectaron objetos con el umbral especificado."

    return imagen_con_cajas, texto

# -------------------------------------------------------------------
# 4. EJEMPLO DE USO (SI SE EJECUTA DIRECTAMENTE)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # --- Configuración de ejemplo ---
    # Cambia esta ruta por la de tu imagen
<<<<<<< HEAD:codigo/modelo.py
    IMAGEN_PRUEBA = "C:\\Users\\aleja\\Documents\\Analizador-Imagen\\imagenes\\mochila.jpg"
=======
    IMAGEN_PRUEBA = r"C:\\Users\\Vespertino\\Documents\\Analizador-Imagen\\imagenes\\mochila2.jpg"
>>>>>>> 56cb3a6c29b6723a65bed673762d9c15e9010679:src/modelo.py

    # Lista de objetos que queremos detectar (puedes modificarla)
    OBJETOS_BUSCAR = ["un gato", "un perro", "una persona", "un coche", "una mochila", "Luffy", "Zoro", "Nami", "Usopp", "Sanji", "Chopper", "Robin", "Franky", "Brook"]

    # Umbral de confianza (ajústalo según necesites)
    UMBRAL = 0.2

    # Llamar a la función principal
    try:
        resultado_img, info = detectar_objetos(IMAGEN_PRUEBA, OBJETOS_BUSCAR, UMBRAL)

        # Mostrar imagen resultante (se abre con el visor predeterminado)
        resultado_img.show()

        # Guardar la imagen con las detecciones en el disco
        nombre_guardado = "resultado_" + os.path.basename(IMAGEN_PRUEBA)
        resultado_img.save(nombre_guardado)
        print(f"Imagen guardada como: {nombre_guardado}")

        # Mostrar información en consola
        print("\n--- DETECCIONES ---")
        print(info)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")

# -------------------------------------------------------------------
# 5. (OPCIONAL) INTERFAZ GRÁFICA CON GRADIO
#    Para usar esta parte, instala gradio: pip install gradio
#    Descomenta las siguientes líneas si quieres probar la interfaz.
# -------------------------------------------------------------------
"""
import gradio as gr

def interfaz_gradio(imagen, objetos_texto, umbral):
    # Convertir el texto de objetos en lista (separados por coma)
    candidatos = [obj.strip() for obj in objetos_texto.split(",") if obj.strip()]
    # Procesar la imagen (imagen viene como objeto PIL)
    # Pero nuestra función espera una ruta, así que guardamos temporalmente
    temp_path = "temp_image.jpg"
    imagen.save(temp_path)
    img_result, texto = detectar_objetos(temp_path, candidatos, umbral)
    os.remove(temp_path)
    return img_result, texto

# Crear interfaz
iface = gr.Interface(
    fn=interfaz_gradio,
    inputs=[
        gr.Image(type="pil", label="Sube tu imagen"),
        gr.Textbox(value="un gato, un perro, una persona, un coche, una mochila",
                   label="Objetos a buscar (separados por coma)"),
        gr.Slider(0.0, 1.0, value=0.1, label="Umbral de confianza")
    ],
    outputs=[
        gr.Image(label="Imagen con detecciones"),
        gr.Textbox(label="Detalle de detecciones")
    ],
    title="Detector de Objetos Zero-Shot",
    description="Utiliza el modelo OWLv2 de Google para detectar objetos en imágenes mediante texto."
)

if __name__ == "__main__":
    iface.launch()
"""
