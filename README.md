# Otestovanie modelu – návod pre oponenta

Tento návod slúži na jednoduché vyskúšanie dvoch skriptov: `yolo_validate.py` a `yolo_predict.py`.

> **Vzorka datasetu a natrénované modely sú odovzdané v AIS.**

---

## Požiadavky

- Python 3.10
- NVIDIA GPU s CUDA 11.8 alebo 12.x *(odporúčané; bez GPU funguje na CPU)*

---

## 1. Klonovanie repozitára

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
```

---

## 2. Inštalácia knižníc

```bash
python -m venv venv

# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python scikit-learn numpy matplotlib seaborn
```

> Pre CPU-only verziu PyTorch viď https://pytorch.org/get-started/locally/

---

## 3. Rozbalenie datasetu a modelov z AIS

Rozbaľte stiahnutý archív z AIS priamo do koreňového priečinka repozitára. Výsledná štruktúra by mala vyzerať takto:

```
<repo>/
├── src/
├── models/
│   ├── full/best.pt
│   └── quater/best.pt
└── dataset/
    ├── dataset_full/val/
    └── dataset_quater/val/
```

---

## 4. Spustenie

### Vyhodnotenie s metrikami (`yolo_validate.py`)

Vypíše accuracy a classification report, uloží anotované obrázky a `predictions.csv`.

```bash
cd src
python yolo_validate.py
```

### Predikcia na obrázkoch (`yolo_predict.py`)

Spustí predikciu a uloží výsledky do `runs/predict/`.

```bash
cd src
python yolo_predict.py
```

> Pre štvrtinový model zmeňte v skriptoch `models/full/best.pt` → `models/quater/best.pt`
> a `dataset/dataset_full/val` → `dataset/dataset_quater/val`.
