from transformers import CLIPProcessor, CLIPModel, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm.auto import tqdm

MODEL_NAME  = "openai/clip-vit-base-patch32"
DATA_DIR    = "dataset/dataset_quater"
NUM_EPOCHS  = 20
BATCH_SIZE  = 32
LR          = 1e-5
SAVE_DIR    = "clip_best_quater"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
print(dataset)

class_names = dataset["train"].features["label"].names
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in id2label.items()}
print("Classes:", id2label)

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

text_prompts = [
    f"a document with {label}" for label in class_names
]

def transform(batch):
    batch["pixel_values"] = processor(
        images=[img.convert("RGB") for img in batch["image"]],
        return_tensors="pt",
        padding=True
    )["pixel_values"]
    return batch

dataset = dataset.with_transform(transform)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

train_loader = DataLoader(dataset["train"],      batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(dataset["test"],       batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}  Test: {len(dataset['test'])}")

base_model = CLIPModel.from_pretrained(MODEL_NAME)

# Custom klasifikačná hlava
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

model = CLIPClassifier(base_model, num_labels=len(id2label))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        logits = model(pixel_values=pixel_values)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            logits = model(pixel_values=pixel_values)
            preds  = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'id2label': id2label,
        }, f"{SAVE_DIR}/best_model.pt")
        print(f"           ↳ Best model saved  (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        logits = model(pixel_values=pixel_values)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4,
))