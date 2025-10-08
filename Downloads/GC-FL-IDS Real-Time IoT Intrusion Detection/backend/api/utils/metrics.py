# backend/api/utils/metrics.py
from typing import Dict, Any
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report

def evaluate(y_true, y_pred) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {
        'accuracy': acc,
        'precision_w': pr,
        'recall_w': rc,
        'f1_w': f1,
        'confusion_matrix': cm.tolist(),
        'report': rep
    }
