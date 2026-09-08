"""
=============================================================================
 FULL MODEL TRAINING CAPTURE SCRIPT
 Runs Phase 3, 4, 5, 6 with complete terminal output captured to one file.
 Every epoch, every model, every metric — nothing filtered.
=============================================================================
"""

import subprocess
import sys
import os
import time
from datetime import datetime

BASE_DIR  = r'c:\studies\sports monitoring system'
OUT_FILE  = os.path.join(BASE_DIR, 'reports', 'full_model_results.txt')
PYTHON    = sys.executable

PHASES = [
    ('PHASE 3 — Data Preprocessing & EDA',         'phase3_eda_preprocessing.py'),
    ('PHASE 4 — Physical Movement ML Models',       'phase4_movement_model.py'),
    ('PHASE 5 — Sensor-Based ML Models',            'phase5_sensor_model.py'),
    ('PHASE 6 — MRI Image Classification (Deep Learning)', 'phase6_mri_classification.py'),
]

DIVIDER  = '=' * 80
DIVIDER2 = '-' * 80

def run_phase(script_name, label, out_fh):
    script_path = os.path.join(BASE_DIR, script_name)
    header = f"""
{DIVIDER}
{DIVIDER}
  {label}
  Script : {script_name}
  Start  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
{DIVIDER}
{DIVIDER}

"""
    out_fh.write(header)
    out_fh.flush()
    print(header, end='')

    t0 = time.time()

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8']       = '1'

    proc = subprocess.Popen(
        [PYTHON, '-X', 'utf8', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace',
        env=env,
    )

    for line in proc.stdout:
        out_fh.write(line)
        out_fh.flush()
        print(line, end='', flush=True)

    proc.wait()
    elapsed = time.time() - t0

    footer = f"""
{DIVIDER2}
  {label}
  End    : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
  Elapsed: {elapsed:.1f} seconds
  Exit   : {proc.returncode}
{DIVIDER2}

"""
    out_fh.write(footer)
    out_fh.flush()
    print(footer, end='')
    return proc.returncode


def main():
    os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

    banner = f"""
{'#' * 80}
#{'':^78}#
#  {'REAL-TIME SPORTS INJURY MONITORING SYSTEM':^74}  #
#  {'COMPLETE MODEL TRAINING & EVALUATION OUTPUT':^74}  #
#{'':^78}#
#  {'Generated: ' + datetime.now().strftime('%Y-%m-%d  %H:%M:%S'):^74}  #
#  {'Python: ' + sys.version.split()[0]:^74}  #
#{'':^78}#
{'#' * 80}

Contents of this file:
  1. PHASE 3  — Data Quality Analysis, EDA, Feature Engineering,
                Feature Selection, Train/Val/Test Split Reports
  2. PHASE 4  — Physical Movement ML Models:
                Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
                (hyperparameter tuning, validation metrics, test metrics,
                 confusion matrices, feature importance, ROC-AUC)
  3. PHASE 5  — Sensor-Based ML Models:
                Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
                (same evaluation suite as Phase 4)
  4. PHASE 6  — MRI Deep Learning (CNN / Transfer Learning):
                SimpleCNN, MobileNetV2, ResNet18, EfficientNet-B0
                (every epoch training loss, validation loss, accuracy,
                 early stopping events, learning rate schedule,
                 test evaluation, confusion matrices, Grad-CAM)

IMPORTANT NOTES:
  - Phase 6 uses PSEUDO-LABELS because the MRI dataset has no real labels.
    The DL metrics in Phase 6 carry NO clinical meaning.
    They validate that the training pipeline runs correctly end-to-end.
  - Phase 3/4/5 metrics are computed on real labelled tabular data.

{'#' * 80}

"""
    with open(OUT_FILE, 'w', encoding='utf-8', errors='replace') as fh:
        fh.write(banner)
        print(banner, end='')

        results = {}
        for label, script in PHASES:
            rc = run_phase(script, label, fh)
            results[label] = ('OK' if rc == 0 else f'FAILED (exit {rc})')

        # Final summary
        summary = f"""
{'#' * 80}
  EXECUTION SUMMARY
{'#' * 80}

"""
        for label, status in results.items():
            summary += f"  {'[' + status + ']':12s}  {label}\n"

        summary += f"""
  Output file: {OUT_FILE}
  Generated  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}

{'#' * 80}
"""
        fh.write(summary)
        print(summary)

    print(f'\n✓ All output saved to:\n  {OUT_FILE}\n')


if __name__ == '__main__':
    main()
