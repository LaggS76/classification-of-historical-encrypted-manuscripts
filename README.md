# classification-of-historical-encrypted-manuscripts

Táto príručka popisuje postup inštalácie a spustenia skriptov elektronickej prílohy.

Zdrojové kódy sú dostupné na:  
https://github.com/<user>/<repo>

---

## 1. Požiadavky

- **Python 3.10**
- **NVIDIA GPU s CUDA 11.8 alebo 12.x** (odporúčané; bez GPU prebieha trénovanie na CPU)
- **Operačný systém:** Linux, macOS alebo Windows 10/11

---

## 2. Inštalácia

Vytvorte virtuálne prostredie a nainštalujte závislosti:

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install torch torchvision --index-url \
    https://download.pytorch.org/whl/cu118

pip install ultralytics transformers datasets \
    scikit-learn numpy tqdm optuna joblib \
    opencv-python matplotlib seaborn Pillow
```

Pre iné verzie CUDA alebo CPU-only inštaláciu PyTorch pozri:  
https://pytorch.org/get-started/locally/

Modely ViT a CLIP (`google/vit-base-patch16-224`, `openai/clip-vit-base-patch32`) sa automaticky stiahnu z Hugging Face Hub pri prvom spustení. Modely z rodiny YOLO sa rovnako automaticky stiahnu prostredníctvom knižnice Ultralytics.

---

## 3. Konfigurácia a spustenie skriptov

Každý skript obsahuje konfiguračné premenné na začiatku súboru (cesta k datasetu, cesta k modelu, počet epoch a pod.), ktoré je potrebné upraviť pred spustením.

### Prehľad skriptov

| Skript | Kľúčové premenné |
|---|---|
| `dataset_distribution.py` | `input_dir`, `output_dir`, pomery rozdelenia (70/20/10) |
| `image_cut.py` | `input_root`, `output_root` |
| `distribution_graph.py` | `dataset_root` |
| `yolo_train.py` | cesta k modelu `yolo11m-cls.pt`, `data`, `epochs`, `batch` |
| `yolo_validate.py` | `MODEL_PATH`, `DATASET_PATH`, `OUTPUT_DIR` |
| `yolo_predict.py` | cesta k modelu, `source` |
| `vit_train.py` | `MODEL_NAME`, `DATA_DIR`, `NUM_EPOCHS`, `BATCH_SIZE`, `LR`, `SAVE_DIR` |
| `vit_validate.py` | `LOAD_MODEL_DIR`, `DATA_DIR` |
| `vlm_train.py` | `MODEL_NAME`, `DATA_DIR`, `NUM_EPOCHS`, `BATCH_SIZE`, `LR`, `SAVE_DIR` |
| `vlm_validate.py` | `LOAD_MODEL_DIR`, `DATA_DIR` |
| `optuna_yolo.py` | cesta k modelu, `data`, `n_trials` |
| `optuna_vit.py` | `MODEL_NAME`, `DATA_DIR` |
| `optuna_vlm.py` | `MODEL_NAME`, `DATA_DIR` |

Každý skript sa spúšťa príkazom:

```bash
python <nazov_skriptu>.py
```

---

## 4. Odporúčaný postup

1. Pripravte dataset pomocou `dataset_distribution.py`  
   (prípadne `image_cut.py` pre štvrtinový dataset).

2. Skontrolujte distribúciu tried pomocou `distribution_graph.py`.

3. Voliteľne spustite optimalizáciu hyperparametrov pomocou Optuna  
   (`optuna_*.py`).

4. Spustite trénovanie zvoleného modelu:
   - `yolo_train.py`
   - `vit_train.py`
   - `vlm_train.py`

5. Vyhodnoťte model príslušným validačným skriptom:
   - `*_validate.py`
```
