from transformers import ViTForImageClassification, ViTImageProcessor, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm.auto import tqdm

MODEL_NAME  = "google/vit-base-patch16-224"
DATA_DIR    = "dataset/dataset_full"
NUM_EPOCHS  = 40
BATCH_SIZE  = 16
LR          = 8.67e-05
SAVE_DIR    = "vit_best_quater"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
print(dataset)

class_names = dataset["train"].features["label"].names
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in id2label.items()}
print("Classes:", id2label)

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

def transform(batch):
    batch["pixel_values"] = processor(
        [img.convert("RGB") for img in batch["image"]],
        return_tensors="pt"
    )["pixel_values"]
    return batch

dataset = dataset.with_transform(transform)

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor([x["label"] for x in batch]),
    }

train_loader = DataLoader(dataset["train"],      batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(dataset["test"],       batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}  Test: {len(dataset['test'])}")

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training steps total: {num_training_steps}")

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
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

            logits = model(pixel_values=pixel_values).logits
            preds  = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
        print(f"           ↳ Best model saved  (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
