import optuna
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import joblib

# Custom klasifikačná hlava pre CLIP
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

# Hľadanie optimálnych hyperparametrov
def objective_clip(trial):
    """
    Objective funkcia pre CLIP - optimalizácia hyperparametrov
    """
    
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    MODEL_NAME = "openai/clip-vit-base-patch32"
    DATA_DIR = "dataset/dataset_full"
    NUM_EPOCHS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    class_names = dataset["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}
    
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
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
    
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    base_model = CLIPModel.from_pretrained(MODEL_NAME)
    model = CLIPClassifier(base_model, num_labels=len(id2label))
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps,
    )
    
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(pixel_values=pixel_values)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(pixel_values=pixel_values)
                preds = logits.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

study_clip = optuna.create_study(
    direction="maximize",
    study_name="clip_cls_optimization"
)

study_clip.optimize(objective_clip, n_trials=15)

print("\n" + "="*60)
print(" NAJLEPŠIE NÁJDENÉ HYPERPARAMETRE (CLIP):")
print("="*60)
for param, value in study_clip.best_params.items():
    print(f"  {param}: {value}")

print(f"\n Najlepšia accuracy: {study_clip.best_value:.4f}")
print(f" Celkový počet trials: {len(study_clip.trials)}")

joblib.dump(study_clip, "optuna_study_clip_full.pkl")
print("\n Štúdia uložená do: optuna_study_clip_full.pkl")

# Trenovanie modelu s optimalizovanými hyperparametrami ---------------------------------------------------------
study_clip = joblib.load("optuna_study_clip_full.pkl")
best_params = study_clip.best_params

print("Trénovanie CLIP s optimálnymi hyperparametrami:")
print("-" * 50)
for param, value in best_params.items():
    print(f"  {param}: {value}")
print("-" * 50 + "\n")

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_DIR = "dataset/dataset_full"
NUM_EPOCHS = 40
SAVE_DIR = "clip_best_full_optuna"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
class_names = dataset["train"].features["label"].names
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in id2label.items()}

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

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

train_loader = DataLoader(dataset["train"], batch_size=best_params['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset["validation"], batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn)

base_model = CLIPModel.from_pretrained(MODEL_NAME)
model = CLIPClassifier(base_model, num_labels=len(id2label))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()

num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
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
            labels = batch["labels"].to(device)
            
            logits = model(pixel_values=pixel_values)
            preds = logits.argmax(dim=-1)
            
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

print("\nTrénovanie dokončené!")
print(f"Model uložený v: {SAVE_DIR}/best_model.pt")
print(f"Najlepšia val accuracy: {best_val_acc:.4f}")