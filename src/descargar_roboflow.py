from roboflow import Roboflow
import os

def main():
    print("Iniciando descarga del dataset desde Roboflow...")
    rf = Roboflow(api_key="hcQjug5LDWstHNaVc8GA")
    project = rf.workspace("aivle5-f7j14").project("one-piece-uuyxt")
    version = project.version(1)
    
    # Descargar en la carpeta actual
    dataset = version.download("yolov8")
    
    print(f"\nDescarga completa.")
    print(f"El dataset de YOLOv8 se encuentra en: {dataset.location}")

if __name__ == "__main__":
    main()
