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
    ├── dataset_full/
        ├── encrypted_text/
        ├── plain_text/
        ├── keys/
        └── mixed/
    └── dataset_quater/
        ├── encrypted_text/
        ├── plain_text/
        ├── keys/
        └── mixed/
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

Spustí predikciu a uloží anotované obrázky do `runs/predict/`.

Pred spustením nastavte v súbore `src/yolo_predict.py` premennú `SOURCE` na priečinok s obrázkami, ktoré chcete predikovať:

```python
# príklady:
SOURCE = "dataset/dataset_full/encrypted_text"  # iba šifrovaný text
SOURCE = "dataset/dataset_full/keys"            # iba kľúče
SOURCE = "dataset/dataset_full/plain_text"      # iba čitateľný text
SOURCE = "dataset/dataset_full/mixed"           # iba zmiešaný obsah
```

> Ak chcete predikovať všetky obrázky naraz, skopírujte ich do jedného priečinka (napr. `dataset/predict/`) a nastavte `SOURCE` na tento priečinok.

```bash
cd src
python yolo_predict.py
```

> Pre štvrtinový model zmeňte `models/full/best.pt` → `models/quater/best.pt`.
