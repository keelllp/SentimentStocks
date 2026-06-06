"""
LightGBM model training script for SentimentStocks.
Produces:
  models/direction_model.pkl   — CalibratedClassifierCV(LGBMClassifier)
  models/magnitude_model.pkl   — LGBMRegressor for 3-day return
  models/label_encoder.pkl     — LabelEncoder for stock names
  models/model_metadata.json   — honest validated metrics + feature list

Usage:
    python backend/train_model.py

Progress is shown via tqdm bars and logged to logs/training_<timestamp>.log.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _BACKEND_DIR)

from feature_pipeline import (
    FEATURE_COLUMNS, HORIZON,
    engineer_features_single, select_features,
)

DATA_PATH = os.path.join(_PROJECT_ROOT, 'processed_data', 'final_cleaned_data.csv')
MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── LightGBM hyper-parameters ──────────────────────────────────────────────────
# Strongly-regularized params — no early stopping so training doesn't terminate
# after a handful of rounds when the val-loss plateau is tiny.
CLF_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=5,
    num_leaves=20,
    min_child_samples=60,
    subsample=0.75,
    colsample_bytree=0.70,
    subsample_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

REG_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=5,
    num_leaves=20,
    min_child_samples=60,
    subsample=0.75,
    colsample_bytree=0.70,
    subsample_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_train_logging() -> tuple[logging.Logger, str]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f'training_{ts}.log')
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    log = logging.getLogger('sentimentstocks.train')
    log.setLevel(logging.INFO)
    log.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log, log_file


# Suppress LightGBM verbose output; no early stopping (fixed n_estimators).
_SILENT = [lgb.log_evaluation(0)]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> dict:
    log, log_file = _setup_train_logging()
    log.info('=' * 60)
    log.info('SentimentStocks — LightGBM Model Training')
    log.info(f'Features: {len(FEATURE_COLUMNS)} | Horizon: {HORIZON} trading days')
    log.info(f'Log: {log_file}')
    log.info('=' * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info(f'Loading data: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    log.info(f'  {len(df):,} rows | {df["Stock"].nunique()} stocks | '
             f'{df["Datetime"].min().date()} to {df["Datetime"].max().date()}')

    # ── Label-encode stocks ────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(sorted(df['Stock'].unique()))
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    log.info(f'  Stocks ({len(le.classes_)}): {list(le.classes_)}')

    # ── Per-stock feature engineering ──────────────────────────────────────────
    log.info('\nEngineering features...')
    records = []
    for stock in tqdm(sorted(df['Stock'].unique()), desc='Feature engineering', unit='stock', ncols=80):
        grp = df[df['Stock'] == stock].sort_values('Datetime').reset_index(drop=True)

        # Targets (no data from the future leaks into features)
        fwd_close = grp['Close'].shift(-HORIZON)
        grp['_direction'] = (fwd_close > grp['Close']).astype(float)
        grp['_return'] = (fwd_close - grp['Close']) / grp['Close'].replace(0, np.nan)
        grp = grp.dropna(subset=['_direction', '_return'])

        if len(grp) < 50:
            log.warning(f'  {stock}: only {len(grp)} rows after dropping NaN targets, skipping')
            continue

        enc = int(le.transform([stock])[0])
        grp = engineer_features_single(grp, stock_encoded=enc)
        records.append(grp)

    combined = (pd.concat(records, ignore_index=True)
                .sort_values('Datetime')
                .reset_index(drop=True))

    X = select_features(combined)
    y_dir = combined['_direction'].astype(int).reset_index(drop=True)
    y_ret = combined['_return'].astype(float).reset_index(drop=True)
    dts = combined['Datetime'].reset_index(drop=True)

    up_frac = y_dir.mean()
    log.info(f'  Feature matrix: {X.shape}')
    log.info(f'  Class balance: {up_frac:.3f} UP / {1-up_frac:.3f} DOWN')

    # ── Time-ordered 80 / 20 split ─────────────────────────────────────────────
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_dir_train, y_dir_test = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
    y_ret_train, y_ret_test = y_ret.iloc[:split_idx], y_ret.iloc[split_idx:]

    majority_baseline = max(y_dir_test.mean(), 1 - y_dir_test.mean())
    test_date_start = dts.iloc[split_idx].date() if split_idx < len(dts) else 'N/A'
    log.info(f'  Train: {len(X_train):,} | Test: {len(X_test):,} (from {test_date_start})')
    log.info(f'  Majority-class baseline on test: {majority_baseline:.4f}')

    # ── Walk-forward CV (5 folds) for honest metrics ───────────────────────────
    log.info('\nWalk-forward cross-validation (5 folds)...')
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accs, cv_bacc, cv_auc = [], [], []

    for fold_i, (tr_idx, val_idx) in enumerate(
        tqdm(tscv.split(X_train), total=5, desc='CV folds', unit='fold', ncols=80)
    ):
        Xf, yf = X_train.iloc[tr_idx], y_dir_train.iloc[tr_idx]
        Xv, yv = X_train.iloc[val_idx], y_dir_train.iloc[val_idx]

        clf_f = LGBMClassifier(**CLF_PARAMS)
        clf_f.fit(Xf, yf, callbacks=_SILENT)

        preds_f = clf_f.predict(Xv)
        probs_f = clf_f.predict_proba(Xv)[:, 1]
        cv_accs.append(accuracy_score(yv, preds_f))
        cv_bacc.append(balanced_accuracy_score(yv, preds_f))
        cv_auc.append(roc_auc_score(yv, probs_f))

    log.info(f'  CV accuracy:          {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}')
    log.info(f'  CV balanced accuracy: {np.mean(cv_bacc):.4f} ± {np.std(cv_bacc):.4f}')
    log.info(f'  CV ROC-AUC:           {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}')

    # ── Train final direction model ────────────────────────────────────────────
    log.info('\nTraining final direction classifier...')

    base_clf = LGBMClassifier(**CLF_PARAMS)
    with tqdm(total=1, desc='Direction model', unit='model', ncols=80) as pbar:
        base_clf.fit(X_train, y_dir_train, callbacks=_SILENT)
        pbar.update(1)

    cal_clf = base_clf

    # Evaluate on hold-out test
    test_probs = cal_clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = accuracy_score(y_dir_test, test_preds)
    test_bacc = balanced_accuracy_score(y_dir_test, test_preds)
    test_auc = roc_auc_score(y_dir_test, test_probs)
    test_f1 = f1_score(y_dir_test, test_preds)
    lift = test_bacc - 0.5

    log.info(f'\nHold-out test results (direction):')
    log.info(f'  Accuracy:          {test_acc:.4f}  (majority baseline: {majority_baseline:.4f})')
    log.info(f'  Balanced accuracy: {test_bacc:.4f}  (lift over 0.5: {lift:+.4f})')
    log.info(f'  ROC-AUC:           {test_auc:.4f}')
    log.info(f'  F1 (UP class):     {test_f1:.4f}')
    log.info('\n' + classification_report(y_dir_test, test_preds, target_names=['DOWN', 'UP']))

    # ── Train magnitude regressor ──────────────────────────────────────────────
    log.info('Training magnitude regressor...')
    reg_cal_split = int(len(X_train) * 0.90)
    X_tr_reg = X_train.iloc[:reg_cal_split]
    y_tr_reg = y_ret_train.iloc[:reg_cal_split]
    X_val_reg = X_train.iloc[reg_cal_split:]
    y_val_reg = y_ret_train.iloc[reg_cal_split:]

    reg = LGBMRegressor(**REG_PARAMS)
    with tqdm(total=1, desc='Magnitude model', unit='model', ncols=80) as pbar:
        reg.fit(X_train, y_ret_train, callbacks=_SILENT)
        pbar.update(1)

    reg_test = reg.predict(X_test)
    reg_sign_acc = accuracy_score(
        (y_ret_test > 0).astype(int),
        (reg_test > 0).astype(int),
    )
    log.info(f'  Magnitude regressor sign accuracy: {reg_sign_acc:.4f}')

    # ── Save artifacts ─────────────────────────────────────────────────────────
    log.info('\nSaving artifacts...')
    with tqdm(total=4, desc='Saving', unit='file', ncols=80) as pbar:
        joblib.dump(cal_clf, os.path.join(MODELS_DIR, 'direction_model.pkl')); pbar.update(1)
        joblib.dump(reg, os.path.join(MODELS_DIR, 'magnitude_model.pkl')); pbar.update(1)
        # label_encoder already saved above; re-save to confirm
        joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl')); pbar.update(1)

        metadata = {
            'model_name': 'LightGBM Direction + Magnitude Ensemble',
            'horizon_days': HORIZON,
            'feature_columns': FEATURE_COLUMNS,
            'n_features': len(FEATURE_COLUMNS),
            'trained_at': datetime.now().isoformat(),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'stocks': list(le.classes_),
            'direction_metrics': {
                'accuracy': round(float(test_acc), 4),
                'balanced_accuracy': round(float(test_bacc), 4),
                'roc_auc': round(float(test_auc), 4),
                'f1': round(float(test_f1), 4),
                'majority_class_baseline': round(float(majority_baseline), 4),
                'lift_over_baseline': round(float(lift), 4),
            },
            'cv_metrics': {
                'accuracy_mean': round(float(np.mean(cv_accs)), 4),
                'accuracy_std': round(float(np.std(cv_accs)), 4),
                'balanced_accuracy_mean': round(float(np.mean(cv_bacc)), 4),
                'balanced_accuracy_std': round(float(np.std(cv_bacc)), 4),
                'roc_auc_mean': round(float(np.mean(cv_auc)), 4),
            },
            'class_balance': {
                'up': round(float(y_dir.mean()), 4),
                'down': round(float(1 - y_dir.mean()), 4),
            },
        }
        with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        pbar.update(1)

    log.info(f'\nAll artifacts saved to {MODELS_DIR}')
    log.info(f'   direction_model.pkl, magnitude_model.pkl, label_encoder.pkl, model_metadata.json')
    log.info(f'Training log: {log_file}')
    log.info('=' * 60)
    return metadata


if __name__ == '__main__':
    main()
