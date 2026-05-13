import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_PATH = "runs/classify/train32/weights/best.pt"            # cesta k modelu
DATASET_PATH = "dataset/dataset_full/val"                       # priečinok s testovacími dátami
OUTPUT_DIR = "predictions/output_yolo11l_full_fixed/val"        # kam sa uložia výsledky
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

image_paths = []
true_labels = []

classes = ["keys", "encrypted_text", "plain_text", "mixed"]

for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(cls_path):
        for img in os.listdir(cls_path):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(cls_path, img))
                true_labels.append(cls)

pred_labels = []
confidences = []

for img_path, true_label in zip(image_paths, true_labels):
    results = model(img_path, verbose=False)
    pred_idx = results[0].probs.top1
    pred_label = results[0].names[pred_idx]
    conf = float(results[0].probs.data[pred_idx])

    pred_labels.append(pred_label)
    confidences.append(conf)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    font_scale = max(0.6, min(w, h) / 800)
    thickness = int(font_scale * 2)
    text = f"{pred_label} ({conf:.2f})"

    color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)

    y_offset = int(h * 0.20)
    x_offset = 25

    cv2.putText(img, text, (x_offset + 3, y_offset + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)

    cv2.putText(img, text, (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    filename = os.path.basename(img_path)
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, img)

csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "true_label", "predicted_label", "confidence"])
    for img, t, p, c in zip(image_paths, true_labels, pred_labels, confidences):
        writer.writerow([os.path.basename(img), t, p, f"{c:.4f}"])

print(f"\n✅ Výsledky uložené do: {csv_path}")
print(f"🖼️ Obrázky s predikciami uložené v: {OUTPUT_DIR}")

acc = accuracy_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels, labels=classes)

print(f"\n=== ŠTATISTIKY ===")
print(f"✅ Accuracy: {acc:.4f}\n")
print("📊 Classification report:")
print(classification_report(true_labels, pred_labels, labels=classes))

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel("Predikovaná trieda")
plt.ylabel("Skutočná trieda")
plt.title("Confusion Matrix – Testovacia množina")
plt.tight_layout()
plt.show()
