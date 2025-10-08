# backend/api/train.py
import os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import resample
# --- FIXED IMPORTS ---
from .utils.preprocess import load_dataset, encode_and_scale, select_features, split_data
from .utils.metrics import evaluate
from .model import cnn_model, rnn_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "RT_IoT2022.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def make_synthetic(n=5000, n_features=20, n_classes=2):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.rand(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = rng.randint(0, n_classes, size=n)
    # introduce slight class imbalance
    y[:int(0.2*n)] = 1
    df = X.copy(); df['label'] = y
    return df, 'label'

def oversample(Xtr, ytr):
    # Reset indexes to avoid alignment issues
    Xtr = Xtr.reset_index(drop=True)
    y_series = pd.Series(ytr).reset_index(drop=True)

    counts = y_series.value_counts()
    max_count = counts.max()

    frames_X, frames_y = [], []
    for cls, cnt in counts.items():
        Xi = Xtr[y_series == cls]
        yi = y_series[y_series == cls]
        if cnt < max_count:
            Xi_up, yi_up = resample(Xi, yi, replace=True, n_samples=max_count, random_state=42)
            frames_X.append(Xi_up)
            frames_y.append(yi_up)
        else:
            frames_X.append(Xi)
            frames_y.append(yi)

    Xb = pd.concat(frames_X).reset_index(drop=True)
    yb = pd.concat(frames_y).reset_index(drop=True)
    perm = np.random.RandomState(42).permutation(len(Xb))
    return Xb.iloc[perm], yb.iloc[perm].values


def main():
    if DATA.exists():
        df, target_col = load_dataset(str(DATA))
    else:
        df, target_col = make_synthetic()

    X_scaled, y, cat_cols = encode_and_scale(df, target_col)
    selected, details = select_features(X_scaled, y, cat_cols, mi_threshold=0.8)
    Xtr, Xte, ytr, yte = split_data(X_scaled[selected], y, test_size=0.2)

    # ----- UNBALANCED -----
    ypred_cnn, note_cnn = cnn_model.train_predict(Xtr, ytr, Xte, dl_ok=True)
    ypred_rnn, note_rnn = rnn_model.train_predict(Xtr, ytr, Xte, dl_ok=True)
    eval_cnn = evaluate(yte, ypred_cnn); eval_cnn["note"] = note_cnn
    eval_rnn = evaluate(yte, ypred_rnn); eval_rnn["note"] = note_rnn

    # ----- BALANCED (oversample) -----
    Xtr_bal, ytr_bal = oversample(Xtr, ytr)
    ypred_cnn_b, note_cnn_b = cnn_model.train_predict(Xtr_bal, ytr_bal, Xte, dl_ok=True)
    ypred_rnn_b, note_rnn_b = rnn_model.train_predict(Xtr_bal, ytr_bal, Xte, dl_ok=True)
    eval_cnn_b = evaluate(yte, ypred_cnn_b); eval_cnn_b["note"] = note_cnn_b
    eval_rnn_b = evaluate(yte, ypred_rnn_b); eval_rnn_b["note"] = note_rnn_b

    # Attack distribution (simple: predicted counts by class id)
    attack_dist = pd.Series(ypred_cnn_b).value_counts().sort_index().to_dict()

    summary = {
        "dataset": "RT-IoT2022" if DATA.exists() else "synthetic",
        "target_col": target_col,
        "feature_selection": {
            "method": details["method"],
            "selected_features": selected,
            "mi_ge_0p8": details["mi_selected_ge_threshold"]  # may be empty
        },
        "unbalanced": { "cnn": eval_cnn, "rnn": eval_rnn },
        "balanced_oversample": { "cnn": eval_cnn_b, "rnn": eval_rnn_b },
        "attack_distribution_pred": attack_dist,
        "last_updated": int(time.time())
    }

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("Saved:", RESULTS_DIR / "metrics.json")

if __name__ == "__main__":
    main()
