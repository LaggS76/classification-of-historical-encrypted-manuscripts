from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_DIR = "dataset/dataset_quater"
BATCH_SIZE = 16
LOAD_MODEL_DIR = "clip_best_quater_optuna"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_labels):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(clip_model.config.projection_dim, num_labels)
        
    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(vision_outputs[1])
        logits = self.classifier(image_embeds)
        return logits

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

print(f"Loading model from: {LOAD_MODEL_DIR}")

checkpoint = torch.load(f"{LOAD_MODEL_DIR}/best_model.pt", map_location=device)
id2label = checkpoint['id2label']
class_names = [id2label[i] for i in sorted(id2label.keys())]
num_labels = len(id2label)

print(f"Classes: {class_names}")
print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")

base_model = CLIPModel.from_pretrained(MODEL_NAME)
loaded_model = CLIPClassifier(base_model, num_labels=num_labels)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.to(device)
loaded_model.eval()

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def transform_loaded(batch):
    batch["pixel_values"] = processor(
        images=[img.convert("RGB") for img in batch["image"]],
        return_tensors="pt",
        padding=True
    )["pixel_values"]
    return batch

dataset_loaded = load_dataset("imagefolder", data_dir=DATA_DIR)
dataset_loaded = dataset_loaded.with_transform(transform_loaded)
test_loader_loaded = DataLoader(dataset_loaded["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader_loaded, desc="Testing loaded model"):
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        logits = loaded_model(pixel_values=pixel_values)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nTest Results for {LOAD_MODEL_DIR}:")
print(classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4,
))