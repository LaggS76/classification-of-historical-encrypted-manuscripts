import optuna
from ultralytics import YOLO
import joblib

# Hľadanie optimálnych hyperparametrov
def objective(trial):
    """
    Objective funkcia pre Optuna - definuje hyperparametre na optimalizáciu
    a vracia metriku (fitness) na minimalizáciu/maximalizáciu
    """
    
    lr0 = trial.suggest_float("lr0", 1e-3, 1e-1, log=True)
    batch = trial.suggest_categorical("batch", [4, 8, 16, 32])
    
    model = YOLO("yolo11m-cls.pt")
    
    results = model.train(
        data="dataset/dataset_quater",
        epochs=30,
        batch=batch,
        lr0=lr0,
        verbose=False,
        plots=False,  
        save=False    
    )
    
    return results.results_dict['metrics/accuracy_top1']

study = optuna.create_study(
    direction="maximize",
    study_name="yolo_cls_optimization"
)

study.optimize(objective, n_trials=20)

print("\n" + "="*60)
print("NAJLEPŠIE NÁJDENÉ HYPERPARAMETRE:")
print("="*60)
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

print(f"\n Najlepšia accuracy: {study.best_value:.4f}")
print(f" Celkový počet trials: {len(study.trials)}")

joblib.dump(study, "optuna_study_quater.pkl")
print("\n Štúdia uložená do: optuna_study_quater.pkl")

# Trenovanie modelu s optimalizovanými hyperparametrami ---------------------------------------------------------
study = joblib.load("optuna_study_quater.pkl")
best_params = study.best_params

print("🏆 Trénovanie s optimálnymi hyperparametrami:")
print("-" * 50)
for param, value in best_params.items():
    print(f"  {param}: {value}")
print("-" * 50 + "\n")

model = YOLO("yolo11m-cls.pt")

results = model.train(
    data="dataset/dataset_quater",
    epochs=200,
    batch=best_params['batch'],
    lr0=best_params['lr0'],
    plots=True,
    save=True,
    verbose=True,
    name="best_hyperparams_final_quater"
)

print("\n Trénovanie dokončené!")
print(f" Model uložený v: runs/classify/best_hyperparams_final_quater/weights/best.pt")
