import os
import matplotlib.pyplot as plt

dataset_root = "dataset/dataset_quater"
splits = ["train", "val", "test"]
valid_ext = (".png", ".jpg", ".jpeg")

data = {}

for split in splits:
    split_path = os.path.join(dataset_root, split)
    class_counts = {}

    for cls in os.listdir(split_path):
        class_path = os.path.join(split_path, cls)

        if not os.path.isdir(class_path):
            continue

        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(valid_ext)
        ])
        class_counts[cls] = count

    data[split] = class_counts

for split in splits:
    classes = list(data[split].keys())
    counts = list(data[split].values())

    plt.figure()
    plt.bar(classes, counts)
    plt.title(f"Zastúpenie tried – {split}")
    plt.xlabel("Trieda")
    plt.ylabel("Počet obrázkov")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()



