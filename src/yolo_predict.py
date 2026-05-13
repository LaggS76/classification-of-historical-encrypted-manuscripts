from ultralytics import YOLO

model = YOLO("models/full/best.pt")  # pre štvrtinovú vzorku: models/quater/best.pt

model.predict(
    source="dataset/dataset_full/val",  # priečinok so vstupnými obrázkami
    save=True,                          # uloží anotované obrázky
    show_conf=True                      # zobrazí skóre spoľahlivosti
)