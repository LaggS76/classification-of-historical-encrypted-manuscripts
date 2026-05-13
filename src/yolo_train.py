from ultralytics import YOLO

model = YOLO("yolo11m-cls.pt")

model.train(
    data="dataset_full",
    epochs=100,
    imgsz=224,
    batch=8
)
