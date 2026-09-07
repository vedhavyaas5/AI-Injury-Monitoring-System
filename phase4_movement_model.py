"""
=============================================================================
 REAL-TIME SPORTS INJURY MONITORING SYSTEM
 PHASE 4 — Physical Movement ML Prediction Model
=============================================================================
 Author  : Sports Injury Monitoring System
 Purpose : Train, tune, evaluate, and save ML models that predict
           injury_risk from biomechanical movement features.
           Output is consumed by Phase 7 (Camera + Pose Estimation)
           and Phase 8 (Risk Fusion Engine).
=============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import warnings
import time

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from scipy.stats import randint, uniform

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline     import Pipeline as ImbPipeline

import xgboost as xgb

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR      = r'c:\studies\sports monitoring system'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
REPORTS_DIR   = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Risk threshold configuration (application-level, NOT clinically validated)
# These values are tunable and should be reviewed by domain experts before
# any clinical or semi-clinical deployment.
LOW_THRESHOLD      = 0.30   # P < 0.30  → LOW risk
MODERATE_THRESHOLD = 0.60   # 0.30 ≤ P < 0.60 → MODERATE risk
                             # P ≥ 0.60  → HIGH risk

# Cross-validation strategy
CV_FOLDS   = 5
RANDOM_STATE = 42

# Styling
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'figure.facecolor': 'white'})
DIVIDER = '─' * 65

def section(title):
    print(f'\n{"═"*65}')
    print(f'  {title}')
    print(f'{"═"*65}')

def sub(title):
    print(f'\n{DIVIDER}')
    print(f'  {title}')
    print(DIVIDER)


# =============================================================================
# SECTION 1 — LOAD PHASE 3 DATASETS
# =============================================================================
section('1. LOADING PHASE 3 PROCESSED DATASETS')

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train_movement.csv'))
X_val   = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_val_movement.csv'))
X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test_movement.csv'))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train_movement.csv')).squeeze()
y_val   = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_val_movement.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test_movement.csv')).squeeze()

FEATURE_COLS = list(X_train.columns)

print(f'  Training set   : {X_train.shape[0]} samples × {X_train.shape[1]} features')
print(f'  Validation set : {X_val.shape[0]} samples × {X_val.shape[1]} features')
print(f'  Test set       : {X_test.shape[0]} samples × {X_test.shape[1]} features')
print(f'  Feature columns: {FEATURE_COLS}')

# Target distribution
sub('Target Distribution — injury_risk')
for split_name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    c = y.value_counts().sort_index()
    p = y.value_counts(normalize=True).sort_index() * 100
    print(f'  {split_name:6s}  Class 0: {c[0]:>4} ({p[0]:4.1f}%)  '
          f'Class 1: {c.get(1,0):>4} ({p.get(1,0.0):4.1f}%)')

# Class-imbalance ratio (training set used for SMOTE decision)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
imbalance_ratio = neg / max(pos, 1)
print(f'\n  Training imbalance ratio: {imbalance_ratio:.1f}:1')
USE_SMOTE = imbalance_ratio > 5
print(f'  SMOTE will be applied: {USE_SMOTE}  '
      f'(threshold > 5:1 — applied to training data only)')

# XGBoost scale_pos_weight = majority / minority
SCALE_POS_WEIGHT = round(neg / max(pos, 1), 2)
print(f'  XGBoost scale_pos_weight: {SCALE_POS_WEIGHT}')


# =============================================================================
# SECTION 2 — PREPROCESSING PIPELINES
# =============================================================================
section('2. PREPROCESSING PIPELINES')
print("""
  Strategy:
  ─────────────────────────────────────────────────────────
  • Logistic Regression, SVM
      → SimpleImputer(median) → StandardScaler
        (distance/gradient-based models require feature scaling)
  • Decision Tree, Random Forest, XGBoost
      → SimpleImputer(median) only
        (tree splits are threshold-based; scaling not required)
  • SMOTE (if enabled)
      → Applied inside imbalanced-learn Pipeline AFTER imputation,
        BEFORE the classifier. Never applied to validation/test data.
  ─────────────────────────────────────────────────────────
""")

# Shared imputer step (fits on training data only)
imputer_step  = ('imputer',  SimpleImputer(strategy='median'))
scaler_step   = ('scaler',   StandardScaler())

def make_scaled_pipeline(classifier):
    """Pipeline: impute → scale → classify (for LR, SVM)."""
    if USE_SMOTE:
        return ImbPipeline([
            imputer_step,
            scaler_step,
            ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=min(pos-1, 5))),
            ('clf',   classifier)
        ])
    return Pipeline([imputer_step, scaler_step, ('clf', classifier)])

def make_tree_pipeline(classifier):
    """Pipeline: impute → classify (for tree-based models)."""
    if USE_SMOTE:
        return ImbPipeline([
            imputer_step,
            ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=min(pos-1, 5))),
            ('clf',   classifier)
        ])
    return Pipeline([imputer_step, ('clf', classifier)])


# =============================================================================
# SECTION 3 — MODEL DEFINITIONS & HYPERPARAMETER GRIDS
# =============================================================================
section('3. MODEL DEFINITIONS & HYPERPARAMETER SEARCH SPACES')

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)

# ── Logistic Regression ───────────────────────────────────────────────────────
lr_base = LogisticRegression(class_weight='balanced', max_iter=2000,
                              random_state=RANDOM_STATE, solver='lbfgs')
lr_grid = {
    'clf__C':       [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
}
lr_search = GridSearchCV(
    make_scaled_pipeline(lr_base), lr_grid,
    cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=0
)

# ── Decision Tree ─────────────────────────────────────────────────────────────
dt_base = DecisionTreeClassifier(class_weight='balanced',
                                  random_state=RANDOM_STATE)
dt_grid = {
    'clf__max_depth':        [3, 5, 7, 10, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf':  [1, 2, 4],
    'clf__criterion':        ['gini', 'entropy'],
}
dt_search = GridSearchCV(
    make_tree_pipeline(dt_base), dt_grid,
    cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=0
)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf_base = RandomForestClassifier(class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_param_dist = {
    'clf__n_estimators':      randint(100, 500),
    'clf__max_depth':         [5, 10, 15, 20, None],
    'clf__min_samples_split': randint(2, 10),
    'clf__min_samples_leaf':  randint(1, 5),
    'clf__max_features':      ['sqrt', 'log2'],
}
rf_search = RandomizedSearchCV(
    make_tree_pipeline(rf_base), rf_param_dist,
    n_iter=30, cv=cv_strategy, scoring='f1',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)

# ── SVM ───────────────────────────────────────────────────────────────────────
svm_base = SVC(class_weight='balanced', probability=True,
               random_state=RANDOM_STATE)
svm_grid = {
    'clf__C':      [0.1, 1, 10, 100],
    'clf__kernel': ['rbf', 'linear'],
    'clf__gamma':  ['scale', 'auto'],
}
svm_search = GridSearchCV(
    make_scaled_pipeline(svm_base), svm_grid,
    cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=0
)

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb_base = xgb.XGBClassifier(
    scale_pos_weight=SCALE_POS_WEIGHT,
    eval_metric='logloss', verbosity=0,
    random_state=RANDOM_STATE, use_label_encoder=False
)
xgb_param_dist = {
    'clf__n_estimators':  randint(100, 400),
    'clf__max_depth':     randint(3, 10),
    'clf__learning_rate': uniform(0.01, 0.3),
    'clf__subsample':     uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__gamma':         uniform(0, 5),
    'clf__reg_alpha':     uniform(0, 1),
    'clf__reg_lambda':    uniform(0.5, 2),
}
xgb_search = RandomizedSearchCV(
    make_tree_pipeline(xgb_base), xgb_param_dist,
    n_iter=40, cv=cv_strategy, scoring='f1',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)

MODELS = {
    'Logistic Regression': lr_search,
    'Decision Tree':       dt_search,
    'Random Forest':       rf_search,
    'SVM':                 svm_search,
    'XGBoost':             xgb_search,
}


# =============================================================================
# SECTION 4 — TRAINING & HYPERPARAMETER TUNING
# =============================================================================
section('4. TRAINING & HYPERPARAMETER TUNING  (CV on training data only)')

# Combine train + val for final evaluation on held-out test set.
# Tuning and early model selection is performed on the validation set.
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

trained_models  = {}
best_params_log = {}

for name, search in MODELS.items():
    sub(f'Training: {name}')
    t0 = time.time()
    search.fit(X_train, y_train)          # tune on training split
    elapsed = time.time() - t0
    trained_models[name] = search.best_estimator_
    best_params_log[name] = search.best_params_
    print(f'  Best CV F1      : {search.best_score_:.4f}')
    print(f'  Best parameters : {search.best_params_}')
    print(f'  Tuning time     : {elapsed:.1f}s')


# =============================================================================
# SECTION 5 — EVALUATION ON VALIDATION & TEST SETS
# =============================================================================
section('5. MODEL EVALUATION')

def evaluate_model(model, X, y, split_name, threshold=0.50):
    """
    Evaluate a fitted pipeline on a given split.
    threshold: decision threshold for the positive class (default 0.5).
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        'Split':     split_name,
        'Accuracy':  round(accuracy_score(y, preds),             4),
        'Precision': round(precision_score(y, preds,
                                           zero_division=0),     4),
        'Recall':    round(recall_score(y, preds,
                                        zero_division=0),        4),
        'F1':        round(f1_score(y, preds,
                                    zero_division=0),            4),
        'ROC-AUC':   round(roc_auc_score(y, proba),              4),
    }

results_val  = []
results_test = []

for name, model in trained_models.items():
    val_metrics  = evaluate_model(model, X_val,  y_val,  'Validation')
    test_metrics = evaluate_model(model, X_test, y_test, 'Test')
    val_metrics['Model']  = name
    test_metrics['Model'] = name
    results_val.append(val_metrics)
    results_test.append(test_metrics)

df_val  = pd.DataFrame(results_val).set_index('Model')
df_test = pd.DataFrame(results_test).set_index('Model')

sub('Validation Set Results')
print(df_val[['Accuracy','Precision','Recall','F1','ROC-AUC']].to_string())

sub('Test Set Results  (final, unseen)')
print(df_test[['Accuracy','Precision','Recall','F1','ROC-AUC']].to_string())

# Detailed classification report for each model on test set
sub('Detailed Classification Reports — Test Set')
for name, model in trained_models.items():
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print(f'\n  ── {name} ──')
    print(classification_report(y_test, preds,
                                target_names=['No Injury (0)', 'Injury (1)'],
                                zero_division=0))


# =============================================================================
# SECTION 6 — BEST MODEL SELECTION
# =============================================================================
section('6. BEST MODEL SELECTION')
print("""
  Selection criterion:
  ─────────────────────────────────────────────────────────
  In an injury-detection context, FALSE NEGATIVES (missing an
  actual injury) are more costly than false positives.
  Primary metric  → Recall (injury class)
  Secondary metric→ F1-score, then ROC-AUC
  Accuracy alone is misleading under class imbalance.
  ─────────────────────────────────────────────────────────
""")

# Score = weighted combination of Recall + F1 + ROC-AUC on validation set
df_val['selection_score'] = (
    df_val['Recall'] * 0.40 +
    df_val['F1']     * 0.35 +
    df_val['ROC-AUC']* 0.25
)
best_model_name = df_val['selection_score'].idxmax()
best_model      = trained_models[best_model_name]

print(f'  Selection scores (Recall×0.4 + F1×0.35 + AUC×0.25):')
print(df_val[['Recall','F1','ROC-AUC','selection_score']].sort_values(
    'selection_score', ascending=False).to_string())
print(f'\n  ✓ BEST MODEL: {best_model_name}')
print(f'    Val  Recall: {df_val.loc[best_model_name,"Recall"]:.4f}')
print(f'    Val  F1    : {df_val.loc[best_model_name,"F1"]:.4f}')
print(f'    Val  AUC   : {df_val.loc[best_model_name,"ROC-AUC"]:.4f}')
print(f'    Test Recall: {df_test.loc[best_model_name,"Recall"]:.4f}')
print(f'    Test F1    : {df_test.loc[best_model_name,"F1"]:.4f}')
print(f'    Test AUC   : {df_test.loc[best_model_name,"ROC-AUC"]:.4f}')


# =============================================================================
# SECTION 7 — CONFUSION MATRICES
# =============================================================================
section('7. CONFUSION MATRICES')

n_models = len(MODELS)
ncols = 3
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
axes = axes.flatten()

for i, (name, model) in enumerate(trained_models.items()):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    cm    = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
        linewidths=0.5,
        xticklabels=['Pred: No Injury', 'Pred: Injury'],
        yticklabels=['True: No Injury', 'True: Injury'],
        annot_kws={'size': 13, 'weight': 'bold'}
    )
    f1  = f1_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    axes[i].set_title(f'{name}\nF1={f1:.3f}  Recall={rec:.3f}',
                      fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Predicted', fontsize=9)
    axes[i].set_ylabel('Actual',    fontsize=9)
    # Highlight best model
    if name == best_model_name:
        for spine in axes[i].spines.values():
            spine.set_edgecolor('#F44336')
            spine.set_linewidth(3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Confusion Matrices — Physical Movement Model (Test Set)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, 'movement_confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print(f'  Saved: movement_confusion_matrix.png')


# =============================================================================
# SECTION 8 — ROC CURVES
# =============================================================================
section('8. ROC CURVES')

fig, ax = plt.subplots(figsize=(9, 7))
colors  = ['#1565C0','#2E7D32','#B71C1C','#6A1B9A','#E65100']

for (name, model), color in zip(trained_models.items(), colors):
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    RocCurveDisplay.from_predictions(
        y_test, proba, ax=ax,
        name=f'{name}  (AUC={auc:.3f})',
        color=color, lw=2
    )

ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5, label='Random baseline')
ax.set_title('ROC Curves — Physical Movement Model (Test Set)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate',  fontsize=11)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'movement_roc_curves.png'), bbox_inches='tight')
plt.close()
print('  Saved: movement_roc_curves.png')


# =============================================================================
# SECTION 9 — FEATURE IMPORTANCE (Random Forest & XGBoost)
# =============================================================================
section('9. FEATURE IMPORTANCE')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, model_name, color in [
    (axes[0], 'Random Forest', '#43A047'),
    (axes[1], 'XGBoost',       '#FB8C00')
]:
    model     = trained_models[model_name]
    # Extract the classifier step from the pipeline
    clf       = model.named_steps['clf']
    imp_vals  = clf.feature_importances_
    imp_series = pd.Series(imp_vals, index=FEATURE_COLS).sort_values()

    bars = ax.barh(imp_series.index, imp_series.values,
                   color=color, edgecolor='white', height=0.6)
    # Annotate values
    for bar, val in zip(bars, imp_series.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.set_title(f'{model_name}\nFeature Importance',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0, imp_series.max() * 1.18)

plt.suptitle('Feature Importance — Physical Movement Model',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fi_path = os.path.join(REPORTS_DIR, 'movement_feature_importance.png')
plt.savefig(fi_path, bbox_inches='tight')
plt.close()
print('  Saved: movement_feature_importance.png')

# Print top 5 for each
for mname in ['Random Forest', 'XGBoost']:
    clf = trained_models[mname].named_steps['clf']
    imp = pd.Series(clf.feature_importances_,
                    index=FEATURE_COLS).sort_values(ascending=False)
    print(f'\n  {mname} — Top 5 features:')
    for feat, val in imp.head(5).items():
        print(f'    {feat:<40} {val:.4f}')


# =============================================================================
# SECTION 10 — MODEL COMPARISON TABLE
# =============================================================================
section('10. MODEL COMPARISON TABLE')

comparison_rows = []
for name in trained_models:
    row = {'Model': name}
    row.update({k: df_test.loc[name, k]
                for k in ['Accuracy','Precision','Recall','F1','ROC-AUC']})
    row['Best_Model'] = '★' if name == best_model_name else ''
    comparison_rows.append(row)

df_comparison = pd.DataFrame(comparison_rows).set_index('Model')
print(df_comparison.to_string())

df_comparison.to_csv(
    os.path.join(REPORTS_DIR, 'movement_model_comparison.csv')
)
print('\n  Saved: movement_model_comparison.csv')

# Visual comparison table
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')
metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']
cell_text = []
row_labels = []
for name in trained_models:
    vals = [f"{df_test.loc[name, m]:.4f}" for m in metrics]
    if name == best_model_name:
        vals = [f"★ {v}" for v in vals]
    cell_text.append(vals)
    row_labels.append(name)

tbl = ax.table(
    cellText=cell_text, rowLabels=row_labels, colLabels=metrics,
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
# Highlight header row
for j in range(len(metrics)):
    tbl[(0, j)].set_facecolor('#1565C0')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
# Highlight best model row
best_row_idx = list(trained_models.keys()).index(best_model_name) + 1
for j in range(len(metrics)):
    tbl[(best_row_idx, j)].set_facecolor('#FFF9C4')

plt.title('Model Comparison — Physical Movement (Test Set)',
          fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'movement_model_comparison_table.png'),
            bbox_inches='tight')
plt.close()
print('  Saved: movement_model_comparison_table.png')


# =============================================================================
# SECTION 11 — SAVE BEST MODEL & PREPROCESSOR
# =============================================================================
section('11. SAVING BEST MODEL & PREPROCESSING OBJECTS')

# Re-fit the best model on train+val combined for maximum training data
print(f'  Re-fitting {best_model_name} on combined train+val ({len(X_trainval)} samples) ...')
best_search_obj = MODELS[best_model_name]
# Build a fresh pipeline with the best hyperparameters (no CV overhead)
final_model = best_search_obj.best_estimator_
final_model.fit(X_trainval, y_trainval)

model_path = os.path.join(MODELS_DIR, 'movement_model.pkl')
joblib.dump(final_model, model_path)
print(f'  Saved: movement_model.pkl')

# Save preprocessor metadata (scaler + feature column list)
preprocessor_meta = {
    'feature_columns': FEATURE_COLS,
    'scaler':          joblib.load(os.path.join(
                           BASE_DIR, 'models', 'preprocessors',
                           'pm_standard_scaler.pkl')),
    'model_name':      best_model_name,
    'low_threshold':   LOW_THRESHOLD,
    'mod_threshold':   MODERATE_THRESHOLD,
    'best_params':     best_params_log[best_model_name],
}
preprocessor_path = os.path.join(MODELS_DIR, 'movement_preprocessor.pkl')
joblib.dump(preprocessor_meta, preprocessor_path)
print(f'  Saved: movement_preprocessor.pkl')


# =============================================================================
# SECTION 12 — PREDICTION FUNCTION
# =============================================================================
section('12. PREDICTION FUNCTION — predict_movement_risk()')

def predict_movement_risk(
    input_data: dict,
    model_path: str = os.path.join(MODELS_DIR, 'movement_model.pkl'),
    meta_path:  str = os.path.join(MODELS_DIR, 'movement_preprocessor.pkl'),
    low_thr:    float = LOW_THRESHOLD,
    mod_thr:    float = MODERATE_THRESHOLD
) -> dict:
    """
    Predict injury risk from biomechanical movement features.

    Parameters
    ----------
    input_data : dict
        Keys must match the feature columns used during training.
        Example:
        {
            'hip_flexion_angle': 85.0,
            'knee_flexion_angle': 110.0,
            'ankle_rotation_angle': -15.0,
            'angular_velocity': 4.5,
            'linear_acceleration': 9.2,
            'ground_reaction_force': 2800.0,
            'postural_instability_index': 0.72,
            'biomechanical_deviation_score': 0.65,
            'fatigue_level': 0.85,
            'movement_intensity': 41.4,
            'biomechanical_risk_index': 0.685,
            'movement_asymmetry': 25.0,
            'fatigue_grf_interaction': 2380.0
        }

    model_path : str
        Path to the serialised model (.pkl).

    meta_path : str
        Path to the serialised preprocessor metadata (.pkl).

    low_thr : float
        Probability below which risk is classified as LOW.
        !! APPLICATION-LEVEL THRESHOLD — NOT clinically validated !!

    mod_thr : float
        Probability below which risk is MODERATE (above low_thr).
        !! APPLICATION-LEVEL THRESHOLD — NOT clinically validated !!

    Returns
    -------
    dict with keys:
        risk_probability_pct : float   — model probability × 100
        risk_category        : str     — 'LOW' | 'MODERATE' | 'HIGH'
        raw_probability      : float   — raw model output [0, 1]
        model_name           : str
        threshold_note       : str
    """
    model = joblib.load(model_path)
    meta  = joblib.load(meta_path)
    feat_cols = meta['feature_columns']

    # Build DataFrame in correct column order
    df_input = pd.DataFrame([input_data])[feat_cols]

    # Probability from model pipeline (includes internal preprocessing)
    raw_prob = float(model.predict_proba(df_input)[0, 1])

    # Risk category using configurable thresholds
    if raw_prob < low_thr:
        category = 'LOW'
    elif raw_prob < mod_thr:
        category = 'MODERATE'
    else:
        category = 'HIGH'

    return {
        'risk_probability_pct': round(raw_prob * 100, 2),
        'risk_category':        category,
        'raw_probability':      round(raw_prob, 6),
        'model_name':           meta['model_name'],
        'threshold_note':       (
            f'LOW < {low_thr*100:.0f}% | '
            f'MODERATE {low_thr*100:.0f}–{mod_thr*100:.0f}% | '
            f'HIGH ≥ {mod_thr*100:.0f}%  '
            '[APPLICATION-LEVEL THRESHOLDS — NOT clinically validated]'
        )
    }


# ── Demo prediction ───────────────────────────────────────────────────────────
sub('Demo Prediction')

# Build a sample from the test set (first injury-positive record)
injury_idx = y_test[y_test == 1].index
if len(injury_idx) > 0:
    sample_row  = X_test.loc[injury_idx[0]].to_dict()
    sample_label = 1
else:
    sample_row  = X_test.iloc[0].to_dict()
    sample_label = int(y_test.iloc[0])

result = predict_movement_risk(sample_row)
print(f'  True label           : {sample_label}')
print(f'  Risk Probability     : {result["risk_probability_pct"]}%')
print(f'  Risk Category        : {result["risk_category"]}')
print(f'  Raw Probability      : {result["raw_probability"]}')
print(f'  Model Used           : {result["model_name"]}')
print(f'  Threshold Note       : {result["threshold_note"]}')

# Run on first 5 test samples
sub('Batch Demo — First 5 Test Samples')
print(f'  {"True":>6}  {"Prob%":>7}  {"Category":>10}')
print(f'  {"-"*30}')
for idx in X_test.index[:5]:
    row = X_test.loc[idx].to_dict()
    res = predict_movement_risk(row)
    true_label = int(y_test.loc[idx])
    print(f'  {true_label:>6}  {res["risk_probability_pct"]:>6.1f}%  '
          f'{res["risk_category"]:>10}')


# =============================================================================
# SECTION 13 — PHASE 7 INTEGRATION EXPLANATION
# =============================================================================
section('13. PHASE 7 INTEGRATION — Real-Time Camera / Pose Estimation')

print("""
  HOW THIS MODEL RECEIVES REAL-TIME DATA IN PHASE 7
  ─────────────────────────────────────────────────────────

  PHASE 7 pipeline (Camera → Pose → Movement Model):

    ┌─────────────────────────────────────────────────────┐
    │  Camera Feed (RGB / Depth)                          │
    │       ↓                                             │
    │  Pose Estimation (MediaPipe / OpenPose / AlphaPose) │
    │       ↓                                             │
    │  Joint Landmark Extraction                          │
    │  (hip, knee, ankle x/y/z coordinates per frame)    │
    │       ↓                                             │
    │  Biomechanical Feature Computation                  │
    │  ┌──────────────────────────────────────────────┐   │
    │  │ hip_flexion_angle  ← angle between vectors   │   │
    │  │ knee_flexion_angle ← knee joint angle        │   │
    │  │ ankle_rotation_angle ← foot orientation      │   │
    │  │ angular_velocity   ← Δangle / Δtime          │   │
    │  │ linear_acceleration ← Δposition / Δtime²     │   │
    │  │ ground_reaction_force ← estimated from mass  │   │
    │  │                          × acceleration      │   │
    │  │ postural_instability_index ← centre-of-mass  │   │
    │  │ biomechanical_deviation_score ← deviation    │   │
    │  │                          from normative gait │   │
    │  │ fatigue_level  ← derived from motion entropy │   │
    │  │ + 4 engineered features from Phase 3         │   │
    │  └──────────────────────────────────────────────┘   │
    │       ↓                                             │
    │  predict_movement_risk(input_data)                  │
    │       ↓                                             │
    │  Risk Probability % + Category (LOW/MOD/HIGH)       │
    │       ↓                                             │
    │  Phase 8 — Risk Fusion Engine                       │
    └─────────────────────────────────────────────────────┘

  Implementation notes for Phase 7:
  • Load movement_model.pkl and movement_preprocessor.pkl ONCE at startup.
  • Compute features on a rolling window (e.g., last 30 frames at 30 fps = 1s).
  • Call predict_movement_risk() per window — not per raw frame.
  • The pipeline's internal scaler handles normalisation automatically.
  • No separate preprocessing step needed at inference time.
""")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section('PHASE 4 COMPLETE — SUMMARY')
print(f"""
  Best Model         : {best_model_name}
  Test Recall        : {df_test.loc[best_model_name, "Recall"]:.4f}
  Test F1            : {df_test.loc[best_model_name, "F1"]:.4f}
  Test ROC-AUC       : {df_test.loc[best_model_name, "ROC-AUC"]:.4f}

  Saved artefacts:
    models/movement_model.pkl
    models/movement_preprocessor.pkl
    reports/movement_model_comparison.csv
    reports/movement_model_comparison_table.png
    reports/movement_confusion_matrix.png
    reports/movement_feature_importance.png
    reports/movement_roc_curves.png
""")
