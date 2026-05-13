import os
from PIL import Image, ImageOps

input_root = "dataset/need_to_cut"
output_root = "dataset_quater"

valid_ext = (".png", ".jpg", ".jpeg")
skip_classes = ["keys"]

for split in ["train", "val", "test"]:
    split_path = os.path.join(input_root, split)

    if not os.path.isdir(split_path):
        print(f"❌ Neexistuje: {split_path}")
        continue

    for cls in os.listdir(split_path):
        class_path = os.path.join(split_path, cls)

        if not os.path.isdir(class_path):
            continue

        if cls in skip_classes:
            print(f"⏭ Preskakujem triedu: {split}/{cls}")
            continue

        # výstupný priečinok
        output_class_path = os.path.join(output_root, split, cls)
        os.makedirs(output_class_path, exist_ok=True)

        files = os.listdir(class_path)
        print(f"✂️ {split}/{cls} → {len(files)} obrázkov")

        for filename in files:
            if not filename.lower().endswith(valid_ext):
                continue

            img_path = os.path.join(class_path, filename)
            img = Image.open(img_path)

            # zachovanie EXIF orientácie
            img = ImageOps.exif_transpose(img)

            w, h = img.size
            half_w, half_h = w // 2, h // 2

            quads = {
                "top_left":     img.crop((0, 0, half_w, half_h)),
                "top_right":    img.crop((half_w, 0, w, half_h)),
                "bottom_left":  img.crop((0, half_h, half_w, h)),
                "bottom_right": img.crop((half_w, half_h, w, h)),
            }

            name, ext = os.path.splitext(filename)

            for key, img_cut in quads.items():
                out_name = f"{name}_{key}{ext}"
                out_path = os.path.join(output_class_path, out_name)
                img_cut.save(out_path)

print("Štvrtiny uložené do dataset_quater so zachovaným train/val/test.")