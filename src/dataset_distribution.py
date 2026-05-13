import os
import shutil
import random

input_dir = "dataset/full"
output_dir = "dataset/final_test"

train_split = 0.7
val_split = 0.2
test_split = 0.1

classes = ["keys", "encrypted_text", "plain_text", "mixed"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    files = os.listdir(os.path.join(input_dir, cls))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    n_test = n_total - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy(
            os.path.join(input_dir, cls, f),
            os.path.join(output_dir, "train", cls, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(input_dir, cls, f),
            os.path.join(output_dir, "val", cls, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(input_dir, cls, f),
            os.path.join(output_dir, "test", cls, f)
        )

    print(f"{cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Dataset bol rozdelený do train / val / test")