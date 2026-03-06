import os
from PIL import Image

corruptas = []
for root, dirs, files in os.walk("./data/raw"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # Verifica integridad
            except Exception as e:
                print(f"Corrupta: {path}")
                corruptas.append(path)

print(f"Total de imágenes corruptas: {len(corruptas)}")
# Si quieres borrarlas:
for c in corruptas:
    os.remove(c)
print("Imágenes corruptas eliminadas.")