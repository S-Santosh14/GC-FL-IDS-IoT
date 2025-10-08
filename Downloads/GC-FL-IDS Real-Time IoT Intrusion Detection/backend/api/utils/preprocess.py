# backend/api/utils/preprocess.py
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    # infer target col (label/class/last)
    candidates = ['label','Label','LABEL','class','Class','attack','Attack','target','Target','y','Y']
    target = None
    for c in candidates:
        if c in df.columns:
            target = c; break
    if target is None:
        target = df.columns[-1]
    return df, target

def encode_and_scale(df: pd.DataFrame, target_col: str):
    y_raw = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    encoders: Dict[str, LabelEncoder] = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    if y_raw.dtype == 'object':
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        if not np.issubdtype(y_raw.dtype, np.integer):
            y = (y_raw > y_raw.median()).astype(int).values
        else:
            y = y_raw.astype(int).values

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, cat_cols

def select_features(X: pd.DataFrame, y: np.ndarray, cat_cols: List[str], mi_threshold: float = 0.8) -> Tuple[List[str], dict]:
    # Mutual Information
    discrete_mask = [c in cat_cols for c in X.columns]
    try:
        mi = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
    except Exception:
        mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi}).sort_values('mi_score', ascending=False)

    mi_selected = mi_df[mi_df['mi_score'] >= mi_threshold]['feature'].tolist()

    # Tree-based (RF) as backup
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

    if mi_selected:
        selected = mi_selected
        method = f"MI>= {mi_threshold}"
    else:
        top_k = min(20, len(rf_imp))
        selected = rf_imp.head(top_k)['feature'].tolist()
        method = f"RF Top {top_k}"

    details = {
        'mi_scores': mi_df.to_dict(orient='records'),
        'mi_selected_ge_threshold': [{'feature': f, 'mi': float(mi_df.loc[mi_df.feature==f,'mi_score'].iloc[0])} for f in mi_selected],
        'rf_importances': rf_imp.to_dict(orient='records'),
        'method': method
    }
    return selected, details

def split_data(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
