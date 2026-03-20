# ============================================================
# MODEL COMPARISON – ALL MODELS
# Project: Team Chemistry in Football
# Output : model_comparison.txt  (3 formatted tables)
#
# Models compared
#   1. Linear Regression          (Step 6)
#   2. Random Forest              (Step 7)
#   3. Gradient Boosting          (Step 7)
#   4. Tuned Random Forest        (Step 8 – GridSearchCV)
#   5. Hybrid 1: GB + SVR         (Step 7-Hybrid)
#   6. Hybrid 2: XGBoost + MLP    (Step 7-Hybrid)
# ============================================================

import io, textwrap
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              confusion_matrix, classification_report,
                              roc_auc_score)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Load data ───────────────────────────────────────────────
train = pd.read_csv("train_set.csv")
test  = pd.read_csv("test_set.csv")

X_train = train.drop(columns=['Chemistry_Index_100'])
y_train = train['Chemistry_Index_100']
X_test  = test.drop(columns=['Chemistry_Index_100'])
y_test  = test['Chemistry_Index_100']

# ── Binning helper ──────────────────────────────────────────
bins   = [-np.inf, 33, 66, np.inf]
labels = ['Low', 'Medium', 'High']
y_test_cls = pd.cut(y_test, bins=bins, labels=labels)

# ── OvR AUC helper (using continuous score as proxy) ────────
def ovr_auc(y_pred_cont, y_true_cls):
    y_bin = label_binarize(y_true_cls, classes=labels)
    scores = [
        -y_pred_cont,
        -np.abs(y_pred_cont - 50),
         y_pred_cont,
    ]
    aucs = []
    for i in range(3):
        try:
            from sklearn.metrics import roc_auc_score as ras
            aucs.append(ras(y_bin[:, i], scores[i]))
        except Exception:
            aucs.append(float('nan'))
    return np.nanmean(aucs)

# ── Train all models ────────────────────────────────────────
print("Training all models … this may take a few minutes.\n")

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("  [1/6] Linear Regression  done")

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("  [2/6] Random Forest       done")

# 3. Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=4, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("  [3/6] Gradient Boosting   done")

# 4. Tuned Random Forest  (best params from step8 grid search — fast re-fit)
rf_tuned = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_split=2,
    min_samples_leaf=1, random_state=42, n_jobs=-1
)
rf_tuned.fit(X_train, y_train)
rf_tuned_pred = rf_tuned.predict(X_test)
print("  [4/6] Tuned Random Forest done")

# 5. Hybrid 1: GB + SVR
svr_pipe  = Pipeline([('sc', StandardScaler()),
                       ('svr', SVR(kernel='rbf', C=10, epsilon=0.5))])
hybrid1   = StackingRegressor(
    estimators=[('gb', GradientBoostingRegressor(n_estimators=100,
                  learning_rate=0.1, max_depth=4, random_state=42)),
                ('svr', svr_pipe)],
    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
hybrid1.fit(X_train, y_train)
h1_pred = hybrid1.predict(X_test)
print("  [5/6] Hybrid 1 (GB+SVR)   done")

# 6. Hybrid 2: XGBoost + MLP
xgb_est = (XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                          random_state=42, verbosity=0, n_jobs=-1) if HAS_XGB
            else GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                           max_depth=4, random_state=42))
mlp_pipe  = Pipeline([('sc', StandardScaler()),
                       ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64),
                                            activation='relu', max_iter=500,
                                            random_state=42, early_stopping=True,
                                            validation_fraction=0.1))])
hybrid2   = StackingRegressor(
    estimators=[('xgb', xgb_est), ('mlp', mlp_pipe)],
    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
hybrid2.fit(X_train, y_train)
h2_pred = hybrid2.predict(X_test)
print("  [6/6] Hybrid 2 (XGB+MLP)  done\n")

# ── Collect metrics for each model ─────────────────────────
model_preds = {
    'Linear Regression':       lr_pred,
    'Random Forest':           rf_pred,
    'Gradient Boosting':       gb_pred,
    'Tuned Random Forest':     rf_tuned_pred,
    'Hybrid 1: GB + SVR':      h1_pred,
    'Hybrid 2: XGB + MLP':     h2_pred,
}

rows = []
for name, pred in model_preds.items():
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    pred_cls = pd.cut(pred, bins=bins, labels=labels)

    # classification metrics (macro avg)
    cr = classification_report(y_test_cls, pred_cls, labels=labels,
                                output_dict=True, zero_division=0)
    prec = cr['macro avg']['precision']
    rec  = cr['macro avg']['recall']
    f1   = cr['macro avg']['f1-score']
    auc_mean = ovr_auc(pred, y_test_cls)

    # confusion matrix → extract diagonal (per-class accuracy)
    cm = confusion_matrix(y_test_cls, pred_cls, labels=labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_cls_acc = np.diag(cm) / cm.sum(axis=1)
        per_cls_acc = np.nan_to_num(per_cls_acc)

    rows.append({
        'Model':      name,
        'RMSE':       round(rmse, 3),
        'MAE':        round(mae,  3),
        'R²':         round(r2,   3),
        'Precision':  round(prec, 3),
        'Recall':     round(rec,  3),
        'F1-Score':   round(f1,   3),
        'Mean AUC':   round(auc_mean, 3),
        'Acc-Low':    round(per_cls_acc[0], 3),
        'Acc-Mid':    round(per_cls_acc[1], 3),
        'Acc-High':   round(per_cls_acc[2], 3),
    })

df = pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════
#  BUILD THREE TABLES
# ════════════════════════════════════════════════════════════

COL = 22   # model name column width
NUM =  9   # numeric column width

def header_line(cols, widths):
    return "| " + " | ".join(f"{c:^{w}}" for c, w in zip(cols, widths)) + " |"

def sep_line(widths):
    return "+-" + "-+-".join("-"*w for w in widths) + "-+"

def data_line(row_data, widths):
    parts = []
    for val, w in zip(row_data, widths):
        s = str(val)
        parts.append(f"{s:<{w}}" if isinstance(val, str) else f"{s:>{w}}")
    return "| " + " | ".join(parts) + " |"

def build_table(title, columns, widths, data_rows):
    lines = []
    total_w = sum(widths) + 3 * (len(widths) - 1) + 4
    lines.append("=" * total_w)
    lines.append(title.center(total_w))
    lines.append("=" * total_w)
    lines.append(sep_line(widths))
    lines.append(header_line(columns, widths))
    lines.append(sep_line(widths))
    for r in data_rows:
        lines.append(data_line(r, widths))
    lines.append(sep_line(widths))
    lines.append("")
    return "\n".join(lines)

# ── TABLE 1: Regression Metrics ─────────────────────────────
t1_cols   = ['Model', 'RMSE', 'MAE', 'R²']
t1_widths = [COL, NUM, NUM, NUM]
t1_rows   = [[r['Model'], r['RMSE'], r['MAE'], r['R²']] for r in rows]
table1    = build_table("TABLE 1 – REGRESSION METRICS (Test Set)", t1_cols, t1_widths, t1_rows)

# ── TABLE 2: Classification Metrics (binned: Low/Medium/High) ─
t2_cols   = ['Model', 'Precision', 'Recall', 'F1-Score', 'Mean AUC']
t2_widths = [COL, NUM, NUM, NUM, NUM]
t2_rows   = [[r['Model'], r['Precision'], r['Recall'], r['F1-Score'], r['Mean AUC']]
             for r in rows]
table2    = build_table("TABLE 2 – CLASSIFICATION METRICS (Macro-Avg, Binned Classes)",
                        t2_cols, t2_widths, t2_rows)

# ── TABLE 3: Per-class Accuracy ──────────────────────────────
t3_cols   = ['Model', 'Acc-Low', 'Acc-Med', 'Acc-High']
t3_widths = [COL, NUM, NUM, NUM]
t3_rows   = [[r['Model'], r['Acc-Low'], r['Acc-Mid'], r['Acc-High']] for r in rows]
table3    = build_table("TABLE 3 – PER-CLASS ACCURACY  (Low <33 | Medium 33-66 | High >66)",
                        t3_cols, t3_widths, t3_rows)

# ── Write output file ────────────────────────────────────────
out_path = "model_comparison.txt"

header_banner = textwrap.dedent("""\
    ============================================================
     PROJECT : Team Chemistry in Football
     REPORT  : All-Model Comparison
     MODELS  : Linear Regression | Random Forest | Gradient
               Boosting | Tuned Random Forest | Hybrid GB+SVR |
               Hybrid XGBoost+MLP
     CLASSES : Low (<33) | Medium (33-66) | High (>66)
    ============================================================

    Three tables are provided:
      Table 1 – Regression metrics      (RMSE, MAE, R²)
      Table 2 – Classification metrics  (Precision, Recall, F1, AUC)
      Table 3 – Per-class accuracy      (Low / Medium / High)

    Best values in each column are starred (*) in the notes below
    each table.
    ============================================================

""")

def best_note(df_col, label, higher_better=True):
    if higher_better:
        idx = df[df_col].idxmax()
    else:
        idx = df[df_col].idxmin()
    return f"  ★ Best {label:12s}: {df.loc[idx,'Model']}  ({df.loc[idx,df_col]})"

notes1 = "\n".join([
    "Notes:",
    best_note('R²',   'R²',   higher_better=True),
    best_note('RMSE', 'RMSE', higher_better=False),
    best_note('MAE',  'MAE',  higher_better=False),
])

notes2 = "\n".join([
    "Notes:",
    best_note('Precision', 'Precision', higher_better=True),
    best_note('Recall',    'Recall',    higher_better=True),
    best_note('F1-Score',  'F1-Score',  higher_better=True),
    best_note('Mean AUC',  'Mean AUC',  higher_better=True),
])

notes3 = "\n".join([
    "Notes:",
    best_note('Acc-Low',  'Acc-Low',  higher_better=True),
    best_note('Acc-Mid',  'Acc-Mid',  higher_better=True),
    best_note('Acc-High', 'Acc-High', higher_better=True),
])

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(header_banner)
    f.write(table1 + "\n" + notes1 + "\n\n\n")
    f.write(table2 + "\n" + notes2 + "\n\n\n")
    f.write(table3 + "\n" + notes3 + "\n")

print("=" * 60)
print(f"Saved → {out_path}")
print("=" * 60)

# Also print to console
with open(out_path, 'r', encoding='utf-8') as f:
    print(f.read())
