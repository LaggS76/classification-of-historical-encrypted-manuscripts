from ultralytics import YOLO

model = YOLO("models/full/best.pt")  # pre štvrtinovú vzorku: models/quater/best.pt

# Nastavte priečinok s obrázkami, ktoré chcete predikovať.
# Príklady:
#   "dataset/dataset_full/encrypted_text"  – iba šifrovaný text
#   "dataset/dataset_full/keys"            – iba kľúče
#   "dataset/dataset_full/plain_text"      – iba čitateľný text
#   "dataset/dataset_full/mixed"           – iba zmiešaný obsah
#   "dataset/predict"                      – vlastný priečinok so ľubovoľnými obrázkami
# Ak chcete predikovať všetky obrázky naraz, skopírujte ich do jedného priečinka dataset/predict.
SOURCE = "dataset/dataset_full/encrypted_text"

model.predict(
    source=SOURCE,
    save=True,       # uloží anotované obrázky do runs/predict/
    show_conf=True   # zobrazí skóre spoľahlivosti
)