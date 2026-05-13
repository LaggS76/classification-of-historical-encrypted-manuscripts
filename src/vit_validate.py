from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

LOAD_MODEL_DIR = "vit_best_quater_optuna"
DATA_DIR = "dataset/dataset_quater"
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor([x["label"] for x in batch]),
    }


print(f"Loading model from: {LOAD_MODEL_DIR}")
loaded_model = ViTForImageClassification.from_pretrained(LOAD_MODEL_DIR)
loaded_processor = ViTImageProcessor.from_pretrained(LOAD_MODEL_DIR)
loaded_model.to(device)
loaded_model.eval()

def transform_loaded(batch):
    batch["pixel_values"] = loaded_processor(
        [img.convert("RGB") for img in batch["image"]],
        return_tensors="pt"
    )["pixel_values"]
    return batch

dataset_loaded = load_dataset("imagefolder", data_dir=DATA_DIR)
dataset_loaded = dataset_loaded.with_transform(transform_loaded)
test_loader_loaded = DataLoader(dataset_loaded["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class_names = list(loaded_model.config.id2label.values())

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader_loaded, desc="Testing loaded model"):
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        logits = loaded_model(pixel_values=pixel_values).logits
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nTest Results for {LOAD_MODEL_DIR}:")
print(classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4,
))