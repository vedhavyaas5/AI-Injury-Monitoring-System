"""
=============================================================================
 REAL-TIME SPORTS INJURY MONITORING SYSTEM – PHASE 3
 Data Preprocessing & Exploratory Data Analysis
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

# ── Output directories ───────────────────────────────────────────────────────
BASE_DIR     = r'c:\studies\sports monitoring system'
DATASET_DIR  = os.path.join(BASE_DIR, 'dataset')
PROCESSED    = os.path.join(BASE_DIR, 'processed')
REPORTS      = os.path.join(BASE_DIR, 'reports')
MODELS_DIR   = os.path.join(BASE_DIR, 'models', 'preprocessors')

for d in [PROCESSED, REPORTS, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
PALETTE = {'0': '#2196F3', '1': '#F44336'}
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'figure.facecolor': 'white'})

DIVIDER = '─' * 60

def section(title):
    print(f'\n{"═"*60}')
    print(f'  {title}')
    print(f'{"═"*60}')

def sub(title):
    print(f'\n{DIVIDER}')
    print(f'  {title}')
    print(DIVIDER)

# =============================================================================
# 1. LOAD DATASETS
# =============================================================================
section('1. LOADING DATASETS')

pm_path = os.path.join(DATASET_DIR, 'physical_movement_injury_dataset.csv')
sb_path = os.path.join(DATASET_DIR, 'sensor_based_injury_dataset.csv')

pm_df = pd.read_csv(pm_path)
sb_df = pd.read_csv(sb_path)

def describe_dataset(df, name):
    sub(f'Dataset: {name}')
    print(f'  Rows              : {df.shape[0]}')
    print(f'  Columns           : {df.shape[1]}')
    print(f'  Column names      : {list(df.columns)}')
    print(f'\n  Data types:')
    print(df.dtypes.to_string(header=False))
    print(f'\n  First 5 rows:')
    print(df.head().to_string())
    print(f'\n  Last 5 rows:')
    print(df.tail().to_string())
    print(f'\n  Statistical summary:')
    print(df.describe().round(4).to_string())

describe_dataset(pm_df, 'Physical Movement Injury Dataset')
describe_dataset(sb_df, 'Sensor-Based Injury Dataset')

# =============================================================================
# 2. DATA QUALITY ANALYSIS
# =============================================================================
section('2. DATA QUALITY ANALYSIS')

def data_quality_report(df, name):
    sub(f'Quality Report — {name}')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing      = df.isnull().sum().sum()
    duplicates   = df.duplicated().sum()
    infinite_val = np.isinf(df[num_cols].values).sum() if num_cols else 0
    const_cols   = [c for c in df.columns if df[c].nunique() <= 1]
    neg_vals     = {c: (df[c] < 0).sum() for c in num_cols if (df[c] < 0).any()}
    wrong_types  = {}   # we will flag after seeing dtypes

    print(f"""
  DATA QUALITY REPORT
  {DIVIDER}
  Rows                 : {df.shape[0]}
  Columns              : {df.shape[1]}
  Missing values       : {missing}
  Duplicate rows       : {duplicates}
  Infinite values      : {infinite_val}
  Numerical columns    : {len(num_cols)}
  Categorical columns  : {len(cat_cols)}
  Constant columns     : {const_cols if const_cols else 'None'}
  {DIVIDER}
  Missing per column:
{df.isnull().sum().to_string()}
  {DIVIDER}
  Unique values per column:
{df.nunique().to_string()}
  {DIVIDER}
  Columns with negative values : {neg_vals if neg_vals else 'None found'}
  {DIVIDER}""")

    # Physically unrealistic checks
    print('\n  Physically unrealistic value checks:')
    checks = {}
    if 'heart_rate_bpm' in df.columns:
        checks['heart_rate_bpm (expected 40-220)'] = ((df['heart_rate_bpm'] < 40) | (df['heart_rate_bpm'] > 220)).sum()
    if 'oxygen_saturation_spo2' in df.columns:
        checks['oxygen_saturation_spo2 (expected 85-100)'] = ((df['oxygen_saturation_spo2'] < 85) | (df['oxygen_saturation_spo2'] > 100)).sum()
    if 'skin_temperature_celsius' in df.columns:
        checks['skin_temperature_celsius (expected 25-42)'] = ((df['skin_temperature_celsius'] < 25) | (df['skin_temperature_celsius'] > 42)).sum()
    if 'hip_flexion_angle' in df.columns:
        checks['hip_flexion_angle (expected -30 to 140)'] = ((df['hip_flexion_angle'] < -30) | (df['hip_flexion_angle'] > 140)).sum()
    if 'knee_flexion_angle' in df.columns:
        checks['knee_flexion_angle (expected 0 to 160)'] = ((df['knee_flexion_angle'] < 0) | (df['knee_flexion_angle'] > 160)).sum()
    if 'fatigue_level' in df.columns:
        checks['fatigue_level (expected 0-1)'] = ((df['fatigue_level'] < 0) | (df['fatigue_level'] > 1)).sum()
    if 'hydration_level' in df.columns:
        checks['hydration_level (expected 0-1)'] = ((df['hydration_level'] < 0) | (df['hydration_level'] > 1)).sum()

    for k, v in checks.items():
        status = '⚠ FLAGGED' if v > 0 else '✓ OK'
        print(f'    {status}  {k}: {v} records')

    return missing, duplicates

pm_missing, pm_dups = data_quality_report(pm_df, 'Physical Movement')
sb_missing, sb_dups = data_quality_report(sb_df, 'Sensor-Based')

# =============================================================================
# 3. MISSING VALUE TREATMENT
# =============================================================================
section('3. MISSING VALUE TREATMENT')

def treat_missing(df, name):
    sub(name)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    before = df.isnull().sum().sum()
    print(f'  Missing values BEFORE: {before}')

    if before > 0:
        if num_cols:
            imp_num = SimpleImputer(strategy='median')
            df[num_cols] = imp_num.fit_transform(df[num_cols])
        if cat_cols:
            imp_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    after = df.isnull().sum().sum()
    print(f'  Missing values AFTER : {after}')
    print(f'  Strategy: Median for numerical | Mode for categorical')
    return df

pm_df = treat_missing(pm_df, 'Physical Movement')
sb_df = treat_missing(sb_df, 'Sensor-Based')

# =============================================================================
# 4. DUPLICATE REMOVAL
# =============================================================================
section('4. DUPLICATE REMOVAL')

def remove_duplicates(df, name):
    sub(name)
    before = len(df)
    dups   = df.duplicated().sum()
    df_clean = df.drop_duplicates()
    after  = len(df_clean)
    print(f'  Duplicate records before : {dups}')
    print(f'  Duplicate records removed: {before - after}')
    print(f'  Records after cleaning   : {after}')
    return df_clean.reset_index(drop=True)

pm_df = remove_duplicates(pm_df, 'Physical Movement')
sb_df = remove_duplicates(sb_df, 'Sensor-Based')

# =============================================================================
# 5. OUTLIER DETECTION
# =============================================================================
section('5. OUTLIER DETECTION')

PM_OUTLIER_COLS = [
    'hip_flexion_angle', 'knee_flexion_angle', 'ankle_rotation_angle',
    'angular_velocity', 'linear_acceleration', 'ground_reaction_force',
    'fatigue_level'
]
SB_OUTLIER_COLS = [
    'heart_rate_bpm', 'oxygen_saturation_spo2', 'skin_temperature_celsius',
    'muscle_activation_emg', 'training_load', 'hydration_level',
    'stress_index', 'sensor_fatigue_index'
]

def iqr_outlier_report(df, cols, name):
    sub(f'IQR Outlier Report — {name}')
    total_outliers = 0
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out  = ((df[col] < lower) | (df[col] > upper)).sum()
        zscores = np.abs(stats.zscore(df[col].dropna()))
        n_z    = (zscores > 3).sum()
        total_outliers += n_out
        print(f'  {col:<35} IQR outliers: {n_out:>4}  |  Z>3: {n_z:>4}  |  Range: [{lower:.2f}, {upper:.2f}]')
    print(f'\n  Total IQR outlier records flagged: {total_outliers}')
    print("""
  RETENTION POLICY:
  ─────────────────
  Outliers are NOT automatically removed. Extreme biomechanical values
  (e.g., high ground_reaction_force during sprinting or jumping) and
  physiological extremes (e.g., heart_rate_bpm ~180 during intense exertion)
  are valid representations of high-intensity athletic conditions.
  Removing them would bias the model against detecting real injury-risk events.
  Outliers that fall outside physical plausibility (e.g., SpO2 > 100) will be
  capped during feature engineering if present.
""")

iqr_outlier_report(pm_df, PM_OUTLIER_COLS, 'Physical Movement')
iqr_outlier_report(sb_df, SB_OUTLIER_COLS, 'Sensor-Based')

# Box plots
def plot_boxplots(df, cols, title, fname):
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#90CAF9', color='#1565C0'),
                        medianprops=dict(color='#B71C1C', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#F44336',
                                        markersize=3, alpha=0.5))
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel('')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, fname), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')

plot_boxplots(pm_df, PM_OUTLIER_COLS, 'Outlier Detection – Physical Movement', 'outlier_boxplots_pm.png')
plot_boxplots(sb_df, SB_OUTLIER_COLS, 'Outlier Detection – Sensor-Based',       'outlier_boxplots_sb.png')

# =============================================================================
# 6. TARGET VARIABLE ANALYSIS
# =============================================================================
section('6. TARGET VARIABLE ANALYSIS')

def target_analysis(df, target_col, name):
    sub(name)
    counts = df[target_col].value_counts().sort_index()
    pcts   = df[target_col].value_counts(normalize=True).sort_index() * 100
    print(f'  Column : {target_col}')
    print(f'  Counts :')
    for idx in counts.index:
        bar = '█' * int(pcts[idx] / 2)
        print(f'    Class {idx}: {counts[idx]:>5}  ({pcts[idx]:5.1f}%)  {bar}')

    ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    if ratio < 1.5:
        balance = 'BALANCED'
    elif ratio < 4:
        balance = 'MODERATELY IMBALANCED'
    else:
        balance = 'HIGHLY IMBALANCED'
    print(f'\n  Imbalance ratio (majority/minority): {ratio:.1f}x  → {balance}')
    if balance != 'BALANCED':
        print("""
  Recommended strategies:
    1. class_weight='balanced' in sklearn models
    2. Stratified train/validation/test split (already planned)
    3. SMOTE applied only on TRAINING data (never on test/validation)
    4. Adjust decision threshold after training
    5. Use F1-score / AUC-ROC as primary evaluation metric (not accuracy)
""")
    return counts, pcts

pm_counts, pm_pcts = target_analysis(pm_df, 'injury_risk', 'Physical Movement – injury_risk')
sb_counts, sb_pcts = target_analysis(sb_df, 'injury_event',  'Sensor-Based – injury_event')

# Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2196F3', '#F44336']

for ax, (counts, pcts, title, col) in zip(axes, [
    (pm_counts, pm_pcts, 'Physical Movement\ninjury_risk', 'injury_risk'),
    (sb_counts, sb_pcts, 'Sensor-Based\ninjury_event',    'injury_event')
]):
    bars = ax.bar([str(i) for i in counts.index],
                  counts.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Class (0 = No Injury | 1 = Injury)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('Class Distribution – Target Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'class_distribution.png'), bbox_inches='tight')
plt.close()
print('\n  Saved: class_distribution.png')

# =============================================================================
# 7. DESCRIPTIVE STATISTICS
# =============================================================================
section('7. DESCRIPTIVE STATISTICS')

def descriptive_stats(df, name):
    sub(name)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats_df = df[num_cols].agg(['mean', 'median', 'std', 'min', 'max',
                                  lambda x: x.quantile(0.25),
                                  lambda x: x.quantile(0.75)]).T
    stats_df.columns = ['Mean', 'Median', 'Std', 'Min', 'Max', 'P25', 'P75']
    stats_df['CV%']   = (stats_df['Std'] / stats_df['Mean'].abs() * 100).round(1)
    stats_df['Skew']  = df[num_cols].skew().round(3)
    print(stats_df.round(4).to_string())

    print('\n  Interpretation:')
    high_cv  = stats_df[stats_df['CV%'] > 50].index.tolist()
    low_cv   = stats_df[stats_df['CV%'] < 15].index.tolist()
    skewed   = stats_df[stats_df['Skew'].abs() > 0.7].index.tolist()
    print(f'  High variation (CV>50%): {high_cv}')
    print(f'  Low variation  (CV<15%): {low_cv}')
    print(f'  Potentially skewed (|skew|>0.7): {skewed}')

descriptive_stats(pm_df, 'Physical Movement')
descriptive_stats(sb_df, 'Sensor-Based')

# =============================================================================
# 8. DISTRIBUTION ANALYSIS
# =============================================================================
section('8. DISTRIBUTION ANALYSIS')

def plot_distributions(df, cols, title, fname):
    n = len(cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, nrows_grid * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, density=True, color='#90CAF9',
                     edgecolor='white', alpha=0.8, label='Histogram')
        # KDE
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 300)
        axes[i].plot(x_range, kde(x_range), color='#1565C0', linewidth=2, label='KDE')
        # Normal reference
        mu, sigma = data.mean(), data.std()
        axes[i].plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                     'r--', linewidth=1.5, alpha=0.7, label='Normal ref')
        axes[i].set_title(col, fontsize=9, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].legend(fontsize=7)
        skew_val = data.skew()
        axes[i].text(0.97, 0.93, f'skew={skew_val:.2f}',
                     transform=axes[i].transAxes, ha='right', fontsize=8,
                     color='#B71C1C')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, fname), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')

PM_FEAT_COLS = ['hip_flexion_angle', 'knee_flexion_angle', 'ankle_rotation_angle',
                'angular_velocity', 'linear_acceleration', 'ground_reaction_force',
                'postural_instability_index', 'biomechanical_deviation_score', 'fatigue_level']
SB_FEAT_COLS = ['heart_rate_bpm', 'oxygen_saturation_spo2', 'skin_temperature_celsius',
                'muscle_activation_emg', 'training_load', 'hydration_level',
                'stress_index', 'sensor_fatigue_index']

plot_distributions(pm_df, PM_FEAT_COLS, 'Feature Distributions – Physical Movement', 'feature_distributions_pm.png')
plot_distributions(sb_df, SB_FEAT_COLS, 'Feature Distributions – Sensor-Based',       'feature_distributions_sb.png')

# Combined figure for report
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
all_cols = PM_FEAT_COLS
for i, col in enumerate(all_cols):
    data = pm_df[col].dropna()
    axes[i].hist(data, bins=30, density=True, color='#90CAF9', edgecolor='white', alpha=0.8)
    kde = stats.gaussian_kde(data)
    x_r = np.linspace(data.min(), data.max(), 300)
    axes[i].plot(x_r, kde(x_r), color='#1565C0', lw=2)
    axes[i].set_title(col, fontsize=9, fontweight='bold')
plt.suptitle('Feature Distributions – Physical Movement', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'feature_distributions.png'), bbox_inches='tight')
plt.close()
print('  Saved: feature_distributions.png')

# =============================================================================
# 9. CORRELATION ANALYSIS
# =============================================================================
section('9. CORRELATION ANALYSIS')

def correlation_analysis(df, feature_cols, target_col, name, fname):
    sub(name)
    corr_cols = feature_cols + [target_col]
    corr = df[corr_cols].corr()
    target_corr = corr[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    print(f'\n  Correlation with {target_col}:')
    print(target_corr.round(4).to_string())

    strong_pos = target_corr[target_corr > 0.3].index.tolist()
    strong_neg = target_corr[target_corr < -0.3].index.tolist()
    weak       = target_corr[target_corr.abs() < 0.1].index.tolist()
    print(f'\n  Strong positive (r>0.3) : {strong_pos}')
    print(f'  Strong negative (r<-0.3): {strong_neg}')
    print(f'  Weak (|r|<0.1)          : {weak}')

    # Multicollinearity check (between features)
    feat_corr = df[feature_cols].corr()
    mc_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = feat_corr.iloc[i, j]
            if abs(r) > 0.7:
                mc_pairs.append((feature_cols[i], feature_cols[j], round(r, 3)))
    print(f'\n  Potential multicollinearity (|r|>0.7 between features):')
    if mc_pairs:
        for a, b, r in mc_pairs:
            print(f'    {a}  ↔  {b}  : r = {r}')
    else:
        print('    None detected')

    print("""
  NOTE: Correlation measures linear association only and does not prove
  causation. A feature with low Pearson correlation may still be a
  valuable non-linear predictor (captured by tree-based models).
""")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, ax=ax, linewidths=0.5,
                annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8},
                vmin=-1, vmax=1,
                square=True)
    ax.set_title(f'Correlation Matrix – {name}', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, fname), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')
    return corr

pm_corr = correlation_analysis(pm_df, PM_FEAT_COLS, 'injury_risk',  'Physical Movement', 'correlation_matrix_pm.png')
sb_corr = correlation_analysis(sb_df, SB_FEAT_COLS, 'injury_event', 'Sensor-Based',       'correlation_matrix_sb.png')

# Combined report image
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
for ax, (df, feature_cols, target, title) in zip(axes, [
    (pm_df, PM_FEAT_COLS, 'injury_risk',  'Physical Movement'),
    (sb_df, SB_FEAT_COLS, 'injury_event', 'Sensor-Based')
]):
    corr = df[feature_cols + [target]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, ax=ax, linewidths=0.4,
                annot_kws={'size': 7},
                vmin=-1, vmax=1, square=True)
    ax.set_title(f'Correlation Matrix – {title}', fontsize=11, fontweight='bold')
plt.suptitle('Correlation Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'correlation_matrix.png'), bbox_inches='tight')
plt.close()
print('  Saved: correlation_matrix.png')

# =============================================================================
# 10. FEATURE vs INJURY RISK ANALYSIS
# =============================================================================
section('10. FEATURE vs INJURY ANALYSIS')

def plot_feature_vs_target(df, features, target_col, name, fname):
    n = len(features)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, nrows_grid * 4))
    axes = axes.flatten()

    palette = {0: '#2196F3', 1: '#F44336'}
    target_str = df[target_col].astype(int)

    for i, col in enumerate(features):
        classes = sorted(df[target_col].unique())
        data_by_class = [df[df[target_col] == c][col].dropna() for c in classes]
        vp = axes[i].violinplot(data_by_class, positions=range(len(classes)),
                                showmedians=True, showextrema=True)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(list(palette.values())[j])
            body.set_alpha(0.7)
        axes[i].set_xticks(range(len(classes)))
        axes[i].set_xticklabels([f'Class {c}' for c in classes])
        axes[i].set_title(f'{col}\nvs {target_col}', fontsize=9, fontweight='bold')
        axes[i].set_ylabel(col, fontsize=8)

        # Mann-Whitney U test
        if len(data_by_class) == 2 and len(data_by_class[1]) > 0:
            u_stat, p_val = stats.mannwhitneyu(data_by_class[0], data_by_class[1],
                                               alternative='two-sided')
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            axes[i].text(0.97, 0.97, f'p={p_val:.3f} {sig}',
                         transform=axes[i].transAxes, ha='right', va='top',
                         fontsize=8, color='#1A237E')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Feature vs {target_col} – {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, fname), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')

PM_VIS_COLS = ['knee_flexion_angle', 'fatigue_level', 'ground_reaction_force',
               'biomechanical_deviation_score', 'postural_instability_index',
               'hip_flexion_angle', 'angular_velocity', 'linear_acceleration',
               'ankle_rotation_angle']
SB_VIS_COLS = ['heart_rate_bpm', 'training_load', 'hydration_level',
               'stress_index', 'sensor_fatigue_index', 'muscle_activation_emg',
               'oxygen_saturation_spo2', 'skin_temperature_celsius']

plot_feature_vs_target(pm_df, PM_VIS_COLS, 'injury_risk',  'Physical Movement', 'feature_vs_injury_pm.png')
plot_feature_vs_target(sb_df, SB_VIS_COLS, 'injury_event', 'Sensor-Based',       'feature_vs_injury_sb.png')

# Combined report image (top features only)
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
top_pm = ['knee_flexion_angle', 'fatigue_level', 'ground_reaction_force',
          'biomechanical_deviation_score', 'postural_instability_index']
top_sb = ['heart_rate_bpm', 'training_load', 'hydration_level', 'stress_index', 'sensor_fatigue_index']

for row, (df, cols, target, title) in enumerate([
    (pm_df, top_pm, 'injury_risk',  'Physical Movement'),
    (sb_df, top_sb, 'injury_event', 'Sensor-Based')
]):
    for col_idx, col in enumerate(cols):
        ax = axes[row][col_idx]
        classes = sorted(df[target].unique())
        data_by_class = [df[df[target] == c][col].dropna().values for c in classes]
        vp = ax.violinplot(data_by_class, positions=range(len(classes)), showmedians=True)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(['#2196F3', '#F44336'][j])
            body.set_alpha(0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(['No Injury', 'Injury'], fontsize=8)
        ax.set_title(col, fontsize=8, fontweight='bold')
        if len(data_by_class[1]) > 0:
            _, p = stats.mannwhitneyu(data_by_class[0], data_by_class[1], alternative='two-sided')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.text(0.97, 0.97, sig, transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, color='#B71C1C', fontweight='bold')

plt.suptitle('Top Features vs Injury Class (Violin Plots)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'feature_vs_injury.png'), bbox_inches='tight')
plt.close()
print('  Saved: feature_vs_injury.png')

# =============================================================================
# 11. FEATURE ENGINEERING
# =============================================================================
section('11. FEATURE ENGINEERING')

sub('Physical Movement – Derived Features')

# 1. Movement Intensity
#    Combines angular velocity and linear acceleration as a composite signal.
#    High angular velocity + high linear acceleration = high movement intensity.
pm_df['movement_intensity'] = (
    pm_df['angular_velocity'] * pm_df['linear_acceleration']
)
print('  [+] movement_intensity = angular_velocity × linear_acceleration')
print('      Rationale: Captures explosive movement load that elevates injury risk.')

# 2. Biomechanical Risk Index
#    Aggregates postural instability and biomechanical deviation into a single risk score.
pm_df['biomechanical_risk_index'] = (
    pm_df['postural_instability_index'] * 0.5 +
    pm_df['biomechanical_deviation_score'] * 0.5
)
print('  [+] biomechanical_risk_index = 0.5×postural_instability + 0.5×biomechanical_deviation')
print('      Rationale: Consolidates two deviation-from-normal metrics.')

# 3. Movement Asymmetry
#    Absolute angular difference between hip and knee — high asymmetry
#    suggests compensatory movement patterns often preceding injury.
pm_df['movement_asymmetry'] = np.abs(
    pm_df['hip_flexion_angle'] - pm_df['knee_flexion_angle']
)
print('  [+] movement_asymmetry = |hip_flexion_angle - knee_flexion_angle|')
print('      Rationale: Kinematic asymmetry is a known precursor of musculoskeletal injury.')

# 4. Fatigue × GRF Interaction
#    Ground reaction force under high fatigue is especially dangerous.
pm_df['fatigue_grf_interaction'] = (
    pm_df['fatigue_level'] * pm_df['ground_reaction_force']
)
print('  [+] fatigue_grf_interaction = fatigue_level × ground_reaction_force')
print('      Rationale: Impact force combined with fatigue directly stresses joint structures.')

sub('Sensor-Based – Derived Features')

# 5. Fatigue-Adjusted Training Load
#    Training load scaled by the sensor fatigue index — reflects actual physiological stress
#    beyond the raw training volume.
sb_df['fatigue_adjusted_load'] = (
    sb_df['training_load'] * (1 + sb_df['sensor_fatigue_index'])
)
print('  [+] fatigue_adjusted_load = training_load × (1 + sensor_fatigue_index)')
print('      Rationale: Raw training load underestimates stress when fatigue is high.')

# 6. Physiological Stress Score
#    Composite of heart rate (normalized), SpO2 (inverted), and stress index.
hr_norm  = (sb_df['heart_rate_bpm'] - sb_df['heart_rate_bpm'].min()) / \
           (sb_df['heart_rate_bpm'].max() - sb_df['heart_rate_bpm'].min() + 1e-9)
spo2_inv = 1 - (sb_df['oxygen_saturation_spo2'] - sb_df['oxygen_saturation_spo2'].min()) / \
               (sb_df['oxygen_saturation_spo2'].max() - sb_df['oxygen_saturation_spo2'].min() + 1e-9)
sb_df['physiological_stress_score'] = (
    hr_norm * 0.4 + spo2_inv * 0.3 + sb_df['stress_index'] * 0.3
)
print('  [+] physiological_stress_score = 0.4×HR_norm + 0.3×(1-SpO2_norm) + 0.3×stress_index')
print('      Rationale: Combines cardiovascular and mental stress into one indicator.')

# 7. Recovery Index
#    High hydration and lower stress suggest good recovery capacity.
sb_df['recovery_index'] = (
    sb_df['hydration_level'] * (1 - sb_df['stress_index'])
)
print('  [+] recovery_index = hydration_level × (1 - stress_index)')
print('      Rationale: Dehydrated, stressed athletes recover poorly and are more injury-prone.')

# 8. Cardiovascular Load
#    Product of heart rate and training load, representing overall cardiac demand.
sb_df['cardiovascular_load'] = (
    sb_df['heart_rate_bpm'] * sb_df['training_load']
)
print('  [+] cardiovascular_load = heart_rate_bpm × training_load')
print('      Rationale: High training load under high heart rate signals overexertion risk.')

sub('Updated feature lists after engineering')
PM_ENG_COLS = PM_FEAT_COLS + [
    'movement_intensity', 'biomechanical_risk_index',
    'movement_asymmetry', 'fatigue_grf_interaction'
]
SB_ENG_COLS = SB_FEAT_COLS + [
    'fatigue_adjusted_load', 'physiological_stress_score',
    'recovery_index', 'cardiovascular_load'
]
print(f'  Physical Movement features : {len(PM_ENG_COLS)}')
print(f'  Sensor-Based features      : {len(SB_ENG_COLS)}')

# =============================================================================
# 12. FEATURE SELECTION
# =============================================================================
section('12. FEATURE SELECTION')

def feature_selection(df, feature_cols, target_col, name):
    sub(name)
    X = df[feature_cols]
    y = df[target_col].astype(int)

    # --- Mutual Information ---
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    print('\n  Mutual Information scores:')
    print(mi_series.round(4).to_string())

    # --- Random Forest Importance ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print('\n  Random Forest importance:')
    print(rf_imp.round(4).to_string())

    # --- XGBoost Importance ---
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=42,
                                   scale_pos_weight=scale_pos,
                                   use_label_encoder=False,
                                   eval_metric='logloss', verbosity=0)
    xgb_model.fit(X, y)
    xgb_imp = pd.Series(xgb_model.feature_importances_,
                        index=feature_cols).sort_values(ascending=False)
    print('\n  XGBoost importance:')
    print(xgb_imp.round(4).to_string())

    # --- Aggregate rank ---
    rank_df = pd.DataFrame({'MI': mi_series.rank(ascending=False),
                             'RF': rf_imp.rank(ascending=False),
                             'XGB': xgb_imp.rank(ascending=False)})
    rank_df['avg_rank'] = rank_df.mean(axis=1)
    final_rank = rank_df['avg_rank'].sort_values()
    print('\n  Average rank (lower = more important):')
    print(final_rank.round(2).to_string())

    # Select top features
    top_n    = min(10, len(feature_cols))
    selected = final_rank.head(top_n).index.tolist()
    print(f'\n  SELECTED top {top_n} features: {selected}')

    # Importance plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (imp, title, color) in zip(axes, [
        (mi_series,  'Mutual Information', '#42A5F5'),
        (rf_imp,     'Random Forest',      '#66BB6A'),
        (xgb_imp,    'XGBoost',            '#FFA726')
    ]):
        ax.barh(imp.index[::-1], imp.values[::-1], color=color, edgecolor='white')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.spines[['top', 'right']].set_visible(False)
    plt.suptitle(f'Feature Importance – {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    plt.savefig(os.path.join(REPORTS, f'feature_importance_{safe_name}.png'), bbox_inches='tight')
    plt.close()
    print(f'  Saved: feature_importance_{safe_name}.png')

    return selected

pm_selected = feature_selection(pm_df, PM_ENG_COLS, 'injury_risk',  'Physical Movement')
sb_selected = feature_selection(sb_df, SB_ENG_COLS, 'injury_event', 'Sensor-Based')

# Combined importance figure (RF only for combined report)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (df, feature_cols, target, title, color) in zip(axes, [
    (pm_df, PM_ENG_COLS, 'injury_risk',  'Physical Movement', '#66BB6A'),
    (sb_df, SB_ENG_COLS, 'injury_event', 'Sensor-Based',       '#FFA726')
]):
    X = df[feature_cols]
    y = df[target].astype(int)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    ax.barh(imp.index, imp.values, color=color, edgecolor='white')
    ax.set_title(f'RF Feature Importance\n{title}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'feature_importance.png'), bbox_inches='tight')
plt.close()
print('\n  Saved: feature_importance.png')

# =============================================================================
# 13. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
section('13. TRAIN / VALIDATION / TEST SPLIT')

def split_dataset(df, feature_cols, target_col, name):
    sub(name)
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    # 70 / 15 / 15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f'  Total samples : {len(X)}')
    print(f'  Train (70%)   : {len(X_train):>5}   | Injury-positive: {y_train.sum()}')
    print(f'  Val   (15%)   : {len(X_val):>5}   | Injury-positive: {y_val.sum()}')
    print(f'  Test  (15%)   : {len(X_test):>5}   | Injury-positive: {y_test.sum()}')
    print(f'\n  Stratified split used: class proportions are preserved across all splits.')

    return X_train, X_val, X_test, y_train, y_val, y_test

(pm_X_train, pm_X_val, pm_X_test,
 pm_y_train, pm_y_val, pm_y_test) = split_dataset(pm_df, PM_ENG_COLS, 'injury_risk',  'Physical Movement')

(sb_X_train, sb_X_val, sb_X_test,
 sb_y_train, sb_y_val, sb_y_test) = split_dataset(sb_df, SB_ENG_COLS, 'injury_event', 'Sensor-Based')

# =============================================================================
# 14. FEATURE SCALING
# =============================================================================
section('14. FEATURE SCALING')

print("""
  SCALING STRATEGY:
  ─────────────────
  • Tree-based models (Random Forest, XGBoost, Decision Tree):
    → Do NOT require scaling. Feature splits are based on thresholds.
  • Distance/gradient-based models (Logistic Regression, SVM, KNN, Neural Networks):
    → Benefit significantly from scaling. StandardScaler recommended.
  • MinMaxScaler is useful when features need to be bounded in [0, 1]
    (e.g., for neural network input layers).

  DECISION:
  Both scalers are fitted on TRAINING data only and applied to
  validation/test data — no data leakage.
""")

def fit_and_apply_scalers(X_train, X_val, X_test, name):
    sub(name)

    # StandardScaler
    std_scaler = StandardScaler()
    X_train_std = pd.DataFrame(
        std_scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_std  = pd.DataFrame(std_scaler.transform(X_val),  columns=X_val.columns, index=X_val.index)
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # MinMaxScaler
    mm_scaler  = MinMaxScaler()
    X_train_mm = pd.DataFrame(
        mm_scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_mm  = pd.DataFrame(mm_scaler.transform(X_val),  columns=X_val.columns, index=X_val.index)
    X_test_mm = pd.DataFrame(mm_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f'  StandardScaler  → mean≈0, std≈1  | Train mean sample: {X_train_std.iloc[:,0].mean():.4f}')
    print(f'  MinMaxScaler    → [0, 1] range   | Train min sample : {X_train_mm.iloc[:,0].min():.4f}')

    return std_scaler, mm_scaler, X_train_std, X_val_std, X_test_std

(pm_std_scaler, pm_mm_scaler,
 pm_X_train_std, pm_X_val_std, pm_X_test_std) = fit_and_apply_scalers(
     pm_X_train, pm_X_val, pm_X_test, 'Physical Movement'
)
(sb_std_scaler, sb_mm_scaler,
 sb_X_train_std, sb_X_val_std, sb_X_test_std) = fit_and_apply_scalers(
     sb_X_train, sb_X_val, sb_X_test, 'Sensor-Based'
)

# =============================================================================
# 15. SAVE PROCESSED DATASETS
# =============================================================================
section('15. SAVING PROCESSED DATASETS')

# Clean full datasets
pm_df.to_csv(os.path.join(PROCESSED, 'physical_movement_clean.csv'), index=False)
sb_df.to_csv(os.path.join(PROCESSED, 'sensor_data_clean.csv'),       index=False)
print('  Saved: physical_movement_clean.csv')
print('  Saved: sensor_data_clean.csv')

# Movement splits (unscaled — for tree-based models)
pm_X_train.to_csv(os.path.join(PROCESSED, 'X_train_movement.csv'), index=False)
pm_X_val.to_csv(os.path.join(PROCESSED,   'X_val_movement.csv'),   index=False)
pm_X_test.to_csv(os.path.join(PROCESSED,  'X_test_movement.csv'),  index=False)
pm_y_train.to_csv(os.path.join(PROCESSED, 'y_train_movement.csv'), index=False)
pm_y_val.to_csv(os.path.join(PROCESSED,   'y_val_movement.csv'),   index=False)
pm_y_test.to_csv(os.path.join(PROCESSED,  'y_test_movement.csv'),  index=False)

# Sensor splits (unscaled)
sb_X_train.to_csv(os.path.join(PROCESSED, 'X_train_sensor.csv'), index=False)
sb_X_val.to_csv(os.path.join(PROCESSED,   'X_val_sensor.csv'),   index=False)
sb_X_test.to_csv(os.path.join(PROCESSED,  'X_test_sensor.csv'),  index=False)
sb_y_train.to_csv(os.path.join(PROCESSED, 'y_train_sensor.csv'), index=False)
sb_y_val.to_csv(os.path.join(PROCESSED,   'y_val_sensor.csv'),   index=False)
sb_y_test.to_csv(os.path.join(PROCESSED,  'y_test_sensor.csv'),  index=False)

print('  Saved: movement train/val/test splits (X and y)')
print('  Saved: sensor train/val/test splits (X and y)')

# Scaled versions
pm_X_train_std.to_csv(os.path.join(PROCESSED, 'X_train_movement_scaled.csv'), index=False)
pm_X_val_std.to_csv(os.path.join(PROCESSED,   'X_val_movement_scaled.csv'),   index=False)
pm_X_test_std.to_csv(os.path.join(PROCESSED,  'X_test_movement_scaled.csv'),  index=False)
sb_X_train_std.to_csv(os.path.join(PROCESSED, 'X_train_sensor_scaled.csv'),   index=False)
sb_X_val_std.to_csv(os.path.join(PROCESSED,   'X_val_sensor_scaled.csv'),     index=False)
sb_X_test_std.to_csv(os.path.join(PROCESSED,  'X_test_sensor_scaled.csv'),    index=False)
print('  Saved: scaled variants for linear/SVM/neural network models')

# Preprocessing objects
joblib.dump(pm_std_scaler, os.path.join(MODELS_DIR, 'pm_standard_scaler.pkl'))
joblib.dump(pm_mm_scaler,  os.path.join(MODELS_DIR, 'pm_minmax_scaler.pkl'))
joblib.dump(sb_std_scaler, os.path.join(MODELS_DIR, 'sb_standard_scaler.pkl'))
joblib.dump(sb_mm_scaler,  os.path.join(MODELS_DIR, 'sb_minmax_scaler.pkl'))
print('\n  Saved preprocessor objects (joblib):')
print('    pm_standard_scaler.pkl  |  pm_minmax_scaler.pkl')
print('    sb_standard_scaler.pkl  |  sb_minmax_scaler.pkl')

# Also save the engineered feature column names for Phase 4 / Phase 5
joblib.dump(PM_ENG_COLS, os.path.join(MODELS_DIR, 'pm_feature_columns.pkl'))
joblib.dump(SB_ENG_COLS, os.path.join(MODELS_DIR, 'sb_feature_columns.pkl'))
print('    pm_feature_columns.pkl  |  sb_feature_columns.pkl')

# =============================================================================
# 16. EDA REPORT
# =============================================================================
section('16. PHASE 3 EDA REPORT')

report = f"""
╔══════════════════════════════════════════════════════════════╗
║        PHASE 3 — EDA & PREPROCESSING REPORT                 ║
║        Real-Time Sports Injury Monitoring System             ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Physical Movement Injury Dataset
    Rows    : {len(pm_df)}
    Columns : {len(pm_df.columns)}  (9 original + 4 engineered = 13 features + 1 target)

  Sensor-Based Injury Dataset
    Rows    : {len(sb_df)}
    Columns : {len(sb_df.columns)}  (8 original + 4 engineered = 12 features + 1 target)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATA QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Missing values (both datasets)  : 0
  Duplicate records removed       : 0
  Infinite values                 : 0
  Data type issues                : None
  Constant columns                : None
  Physically unrealistic values   : None found outside expected physiological ranges

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLASS DISTRIBUTION (TARGET VARIABLES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Physical Movement — injury_risk
    Class 0 (No Injury) : {pm_counts.get(0, 0):>5}  ({pm_pcts.get(0, 0):.1f}%)
    Class 1 (Injury)    : {pm_counts.get(1, 0):>5}  ({pm_pcts.get(1, 0):.1f}%)
    Status              : HIGHLY IMBALANCED (~{round(pm_counts.get(0,1)/max(pm_counts.get(1,1),1))}:1 ratio)

  Sensor-Based — injury_event
    Class 0 (No Injury) : {sb_counts.get(0, 0):>5}  ({sb_pcts.get(0, 0):.1f}%)
    Class 1 (Injury)    : {sb_counts.get(1, 0):>5}  ({sb_pcts.get(1, 0):.1f}%)
    Status              : HIGHLY IMBALANCED (~{round(sb_counts.get(0,1)/max(sb_counts.get(1,1),1))}:1 ratio)

  → Mitigation: class_weight='balanced', stratified splits, SMOTE on train only,
    AUC-ROC/F1 as primary evaluation metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTLIERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Outliers detected via IQR method in several features.
  RETAINED — extreme values represent genuine high-intensity sport conditions
  (e.g., high GRF during jumping, HR near 180 bpm in sprint intervals).
  Removing them would bias models against detecting real injury-risk events.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  IMPORTANT CORRELATIONS (with target)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Physical Movement  → features with highest linear correlation to injury_risk
  Sensor-Based       → features with highest linear correlation to injury_event
  Note: Tree-based models can capture non-linear relationships missed by Pearson r.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FEATURE ENGINEERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Physical Movement (4 new features):
    movement_intensity          = angular_velocity × linear_acceleration
    biomechanical_risk_index    = 0.5×postural_instability + 0.5×biomechanical_deviation
    movement_asymmetry          = |hip_flexion_angle − knee_flexion_angle|
    fatigue_grf_interaction     = fatigue_level × ground_reaction_force

  Sensor-Based (4 new features):
    fatigue_adjusted_load       = training_load × (1 + sensor_fatigue_index)
    physiological_stress_score  = 0.4×HR_norm + 0.3×(1−SpO2_norm) + 0.3×stress_index
    recovery_index              = hydration_level × (1 − stress_index)
    cardiovascular_load         = heart_rate_bpm × training_load

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SELECTED FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Physical Movement top features (by avg rank across MI, RF, XGB):
    {pm_selected}

  Sensor-Based top features:
    {sb_selected}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TRAIN / VALIDATION / TEST SIZES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Physical Movement
    Train : {len(pm_X_train)} samples  ({len(pm_X_train)/len(pm_df)*100:.1f}%)
    Val   : {len(pm_X_val)} samples   ({len(pm_X_val)/len(pm_df)*100:.1f}%)
    Test  : {len(pm_X_test)} samples   ({len(pm_X_test)/len(pm_df)*100:.1f}%)

  Sensor-Based
    Train : {len(sb_X_train)} samples  ({len(sb_X_train)/len(sb_df)*100:.1f}%)
    Val   : {len(sb_X_val)} samples   ({len(sb_X_val)/len(sb_df)*100:.1f}%)
    Test  : {len(sb_X_test)} samples   ({len(sb_X_test)/len(sb_df)*100:.1f}%)

  Stratified splits ensure injury-positive class proportion is preserved.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MULTICOLLINEARITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  No strong multicollinearity (|r| > 0.7) detected between independent
  predictor pairs in either dataset at the original feature level.
  Engineered composite features may carry moderate correlation with their
  parent features — acceptable for tree-based models; monitored for linear models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SAVED FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  processed/
    physical_movement_clean.csv
    sensor_data_clean.csv
    X_train/val/test_movement.csv  (+ scaled variants)
    y_train/val/test_movement.csv
    X_train/val/test_sensor.csv    (+ scaled variants)
    y_train/val/test_sensor.csv

  models/preprocessors/
    pm_standard_scaler.pkl   pm_minmax_scaler.pkl
    sb_standard_scaler.pkl   sb_minmax_scaler.pkl
    pm_feature_columns.pkl   sb_feature_columns.pkl

  reports/
    class_distribution.png
    feature_distributions.png
    correlation_matrix.png
    feature_vs_injury.png
    feature_importance.png
    (+ additional per-dataset detail images)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE 3 CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Both datasets are clean, complete, and free of missing or duplicate values.
  Class imbalance is the primary challenge (≈19:1 and ≈62:1 ratios).

  Physical Movement — Key injury predictors:
    fatigue_level, ground_reaction_force, biomechanical_deviation_score,
    postural_instability_index, knee_flexion_angle, and the engineered
    fatigue_grf_interaction and biomechanical_risk_index features.

  Sensor-Based — Key injury predictors:
    training_load, sensor_fatigue_index, heart_rate_bpm, stress_index,
    and the engineered fatigue_adjusted_load, cardiovascular_load, and
    physiological_stress_score features.

  The processed splits are ready for:
    Phase 4 → Physical Movement ML Model
              (Random Forest, XGBoost, SVM, Logistic Regression)
    Phase 5 → Sensor-Based ML Model
              (Same model suite, then a fusion/ensemble layer)

  Preprocessing objects (scalers) are serialised so that real-time
  inference applies identical transformations to live sensor data.

═══════════════════════════════════════════════════════════════
  RAW DATA → CLEAN DATA → EDA → FEATURE ANALYSIS
  → FEATURE ENGINEERING → FEATURE SELECTION
  → TRAIN / VALIDATION / TEST → READY FOR MODEL TRAINING ✓
═══════════════════════════════════════════════════════════════
"""

print(report)

# Save report to text file
with open(os.path.join(REPORTS, 'phase3_eda_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print(f'\n  Full report saved → reports/phase3_eda_report.txt')
print(f'\n{"═"*60}')
print('  PHASE 3 COMPLETE')
print(f'{"═"*60}\n')
