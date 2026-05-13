from ultralytics import YOLO

model = YOLO("runs/classify/train18/weights/best.pt")

model.predict(
    source="dataset/predict/quater",  
    save=True,              
    show_conf=True          
)