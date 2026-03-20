# ============================================================
# STEP 7 HYBRID: STACKED ENSEMBLE MODELS
# Project: Team Chemistry in Football
# Model 1: Gradient Boosting + SVR (stacked)
# Model 2: XGBoost + MLP Neural Network (stacked)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print("⚠  xgboost not installed. Install with: pip install xgboost")
    print("   Falling back to GradientBoostingRegressor for Model 2 base learner 1.")
    HAS_XGB = False

# ─── 1. Load train & test sets ──────────────────────────────────────
train = pd.read_csv("train_set.csv")
test  = pd.read_csv("test_set.csv")

X_train = train.drop(columns=['Chemistry_Index_100'])
y_train = train['Chemistry_Index_100']

X_test  = test.drop(columns=['Chemistry_Index_100'])
y_test  = test['Chemistry_Index_100']

features = X_train.columns.tolist()

# ─── Helper: bin continuous predictions into 3 classes ──────────────
bins   = [-np.inf, 33, 66, np.inf]
labels = ['Low', 'Medium', 'High']

y_test_cls = pd.cut(y_test, bins=bins, labels=labels)

# ════════════════════════════════════════════════════════════════════
#  HYBRID MODEL 1: Gradient Boosting + SVR  →  Ridge meta-learner
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("HYBRID MODEL 1: Gradient Boosting + SVR (Stacking)")
print("="*60)

# SVR needs scaled features — wrap in a Pipeline
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, epsilon=0.5))
])

gb_estimator = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
)

hybrid1 = StackingRegressor(
    estimators=[
        ('gb',  gb_estimator),
        ('svr', svr_pipeline),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

print("Training Hybrid 1 (GB + SVR)... (includes 5-fold CV stacking)")
hybrid1.fit(X_train, y_train)

h1_pred_train = hybrid1.predict(X_train)
h1_pred_test  = hybrid1.predict(X_test)

h1_train_rmse = np.sqrt(mean_squared_error(y_train, h1_pred_train))
h1_test_rmse  = np.sqrt(mean_squared_error(y_test,  h1_pred_test))
h1_train_r2   = r2_score(y_train, h1_pred_train)
h1_test_r2    = r2_score(y_test,  h1_pred_test)

print(f"\n{'Metric':<20} {'Train':>10} {'Test':>10}")
print("-" * 42)
print(f"{'RMSE':<20} {h1_train_rmse:>10.3f} {h1_test_rmse:>10.3f}")
print(f"{'R²':<20} {h1_train_r2:>10.3f} {h1_test_r2:>10.3f}")

# ════════════════════════════════════════════════════════════════════
#  HYBRID MODEL 2: XGBoost + MLP  →  Ridge meta-learner
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("HYBRID MODEL 2: XGBoost + MLP Neural Network (Stacking)")
print("="*60)

xgb_estimator = (
    XGBRegressor(n_estimators=100, learning_rate=0.1,
                 max_depth=4, random_state=42,
                 verbosity=0, n_jobs=-1)
    if HAS_XGB
    else GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    )
)

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp',    MLPRegressor(hidden_layer_sizes=(128, 64),
                            activation='relu',
                            max_iter=500,
                            random_state=42,
                            early_stopping=True,
                            validation_fraction=0.1))
])

hybrid2 = StackingRegressor(
    estimators=[
        ('xgb', xgb_estimator),
        ('mlp', mlp_pipeline),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

print("Training Hybrid 2 (XGBoost + MLP)... (includes 5-fold CV stacking)")
hybrid2.fit(X_train, y_train)

h2_pred_train = hybrid2.predict(X_train)
h2_pred_test  = hybrid2.predict(X_test)

h2_train_rmse = np.sqrt(mean_squared_error(y_train, h2_pred_train))
h2_test_rmse  = np.sqrt(mean_squared_error(y_test,  h2_pred_test))
h2_train_r2   = r2_score(y_train, h2_pred_train)
h2_test_r2    = r2_score(y_test,  h2_pred_test)

print(f"\n{'Metric':<20} {'Train':>10} {'Test':>10}")
print("-" * 42)
print(f"{'RMSE':<20} {h2_train_rmse:>10.3f} {h2_test_rmse:>10.3f}")
print(f"{'R²':<20} {h2_train_r2:>10.3f} {h2_test_r2:>10.3f}")

# ════════════════════════════════════════════════════════════════════
#  COMBINED SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("HYBRID MODEL COMPARISON SUMMARY")
print("="*60)
summary = pd.DataFrame([
    {'Model': 'Hybrid 1: GB + SVR',  'Train RMSE': round(h1_train_rmse, 3),
     'Test RMSE': round(h1_test_rmse, 3), 'Train R²': round(h1_train_r2, 3),
     'Test R²': round(h1_test_r2, 3)},
    {'Model': 'Hybrid 2: XGB + MLP', 'Train RMSE': round(h2_train_rmse, 3),
     'Test RMSE': round(h2_test_rmse, 3), 'Train R²': round(h2_train_r2, 3),
     'Test R²': round(h2_test_r2, 3)},
])
print(summary.to_string(index=False))

# ════════════════════════════════════════════════════════════════════
#  VISUALISATIONS + CONFUSION MATRIX + ROC  (both models)
#  5 graphs per model + Confusion Matrix + ROC Curve + Error Rate
# ════════════════════════════════════════════════════════════════════
from sklearn.model_selection import learning_curve as lc_fn

hybrid_models = {
    'Hybrid 1: GB + SVR':   (hybrid1, h1_pred_test, '#e67e22', ['GB',  'SVR']),
    'Hybrid 2: XGB + MLP':  (hybrid2, h2_pred_test, '#2980b9', ['XGB', 'MLP']),
}

for idx, (name, (model, y_pred_test, accent, base_names)) in enumerate(hybrid_models.items()):
    short     = "H1" if idx == 0 else "H2"
    residuals = y_test - y_pred_test
    y_pred_cls = pd.cut(y_pred_test, bins=bins, labels=labels)

    # ── Get individual base-model predictions from fitted stacker ────
    base1_pred = model.estimators_[0].predict(X_test)
    base2_pred = model.estimators_[1].predict(X_test)

    # ════════════════════════════════════════════════════════════════
    #  MAIN FIGURE: 5 graphs (3 top row + 2 bottom row)
    # ════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"{name} – 5 Graphical Representations",
                 fontsize=17, fontweight='bold', y=1.02)

    # Use GridSpec: 2 rows, 6 columns → merge cols for layout
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.45)

    # ── Graph 1 (top-left): Actual vs Predicted ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.scatter(y_test, y_pred_test, alpha=0.55,
                edgecolors=accent, facecolors='#ffefd5', s=60, label='Test points')
    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax1.plot(lims, lims, 'r--', lw=1.8, label='Perfect fit')
    ax1.set_xlabel("Actual Chemistry Index", fontsize=10)
    ax1.set_ylabel("Predicted Chemistry Index", fontsize=10)
    ax1.set_title("1. Actual vs Predicted (Test Set)", fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── Graph 2 (top-center): Residual Distribution ──────────────────
    ax2 = fig.add_subplot(gs[0, 2:4])
    sns.histplot(residuals, kde=True, color=accent, ax=ax2, bins=20)
    ax2.axvline(0, color='red', linestyle='--', lw=1.5)
    ax2.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("2. Residual Distribution (Test Set)", fontweight='bold')

    # ── Graph 3 (top-right): Base Model Predictions Comparison ───────
    ax3 = fig.add_subplot(gs[0, 4:6])
    sc = ax3.scatter(base1_pred, base2_pred, c=y_test.values,
                     cmap='RdYlGn', alpha=0.65, edgecolors='gray',
                     s=55, linewidths=0.4)
    plt.colorbar(sc, ax=ax3, label='Actual Chemistry Index')
    ax3.set_xlabel(f"{base_names[0]} Predictions", fontsize=10)
    ax3.set_ylabel(f"{base_names[1]} Predictions", fontsize=10)
    ax3.set_title(f"3. Base-Model Predictions\n({base_names[0]} vs {base_names[1]})",
                  fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ── Graph 4 (bottom-left): Predicted vs Residual ─────────────────
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax4.scatter(y_pred_test, residuals, alpha=0.55,
                edgecolors=accent, facecolors='lightskyblue', s=55)
    ax4.axhline(0, color='red', linestyle='--', lw=1.5)
    ax4.set_xlabel("Predicted Chemistry Index", fontsize=10)
    ax4.set_ylabel("Residual (Actual − Predicted)", fontsize=10)
    ax4.set_title("4. Predicted vs Residual Plot\n(Heteroscedasticity Check)",
                  fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # ── Graph 5 (bottom-right): Cumulative Error Distribution ─────────
    ax5 = fig.add_subplot(gs[1, 3:6])
    abs_errors = np.sort(np.abs(residuals))
    cdf        = np.arange(1, len(abs_errors) + 1) / len(abs_errors)
    ax5.plot(abs_errors, cdf, color=accent, lw=2.2)
    ax5.axvline(np.median(np.abs(residuals)), color='red', linestyle='--',
                lw=1.5, label=f'Median AE = {np.median(np.abs(residuals)):.2f}')
    ax5.axvline(np.mean(np.abs(residuals)), color='purple', linestyle=':',
                lw=1.5, label=f'Mean AE = {np.mean(np.abs(residuals)):.2f}')
    ax5.fill_between(abs_errors, cdf, alpha=0.12, color=accent)
    ax5.set_xlabel("Absolute Error", fontsize=10)
    ax5.set_ylabel("Cumulative Proportion", fontsize=10)
    ax5.set_title("5. Cumulative Error Distribution (CDF)\n(% of test points below error threshold)",
                  fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname_vis = f"step7_{short}_visualisations.png"
    plt.savefig(fname_vis, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n{name} – 5 graphs saved → {fname_vis}")

    # ── Confusion Matrix ─────────────────────────────────────────────
    cm = confusion_matrix(y_test_cls, y_pred_cls, labels=labels)
    fig2, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='YlOrRd' if idx == 0 else 'Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_xlabel("Predicted Class",  fontsize=11)
    ax.set_ylabel("Actual Class",     fontsize=11)
    ax.set_title(f"{name} – Confusion Matrix\n(Low <33 | Medium 33–66 | High >66)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname_cm = f"step7_{short}_confusion_matrix.png"
    plt.savefig(fname_cm, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved → {fname_cm}")

    print(f"\nClassification Report – {name} (binned classes):")
    print(classification_report(y_test_cls, y_pred_cls,
                                labels=labels, zero_division=0))

    # ── ROC Curve (One-vs-Rest) ───────────────────────────────────────
    y_test_bin = label_binarize(y_test_cls, classes=labels)
    class_scores = [
        -y_pred_test,
        -np.abs(y_pred_test - 50),
         y_pred_test,
    ]
    roc_colors = ['#e74c3c', '#8e44ad', '#27ae60']

    fig3, ax_roc = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], class_scores[i])
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=roc_colors[i], lw=2,
                    label=f"{cls} (AUC = {roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.4, label='Random classifier')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=12)
    ax_roc.set_title(f"{name} – ROC Curve (One-vs-Rest)\n"
                     f"(Low <33 | Medium 33–66 | High >66)",
                     fontsize=12, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fname_roc = f"step7_{short}_roc_curve.png"
    plt.savefig(fname_roc, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved → {fname_roc}")

    # ── Error Rate Graph (Learning Curve – RMSE vs training size) ────
    all_X = pd.concat([X_train, X_test])
    all_y = pd.concat([y_train, y_test])

    # Re-create the same stacker (unfitted) for learning_curve
    if idx == 0:
        lc_model = StackingRegressor(
            estimators=[
                ('gb',  GradientBoostingRegressor(n_estimators=100,
                         learning_rate=0.1, max_depth=4, random_state=42)),
                ('svr', Pipeline([('scaler', StandardScaler()),
                                  ('svr', SVR(kernel='rbf', C=10, epsilon=0.5))])),
            ],
            final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    else:
        xgb_lc = (XGBRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=4, random_state=42,
                                verbosity=0, n_jobs=-1) if HAS_XGB
                  else GradientBoostingRegressor(n_estimators=100,
                       learning_rate=0.1, max_depth=4, random_state=42))
        lc_model = StackingRegressor(
            estimators=[
                ('xgb', xgb_lc),
                ('mlp', Pipeline([('scaler', StandardScaler()),
                                  ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64),
                                          activation='relu', max_iter=300,
                                          random_state=42))])),
            ],
            final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)

    train_sizes, train_sc, test_sc = lc_fn(
        lc_model, all_X, all_y,
        cv=3,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.2, 1.0, 6),
        n_jobs=-1
    )
    tr_rmse  = np.sqrt(-train_sc.mean(axis=1))
    tst_rmse = np.sqrt(-test_sc.mean(axis=1))
    tr_std   = np.sqrt(train_sc.std(axis=1))
    tst_std  = np.sqrt(test_sc.std(axis=1))

    fig4, ax_err = plt.subplots(figsize=(8, 5))
    ax_err.plot(train_sizes, tr_rmse, 'o-', color=accent,
                lw=2, label='Train RMSE')
    ax_err.fill_between(train_sizes,
                        tr_rmse  - tr_std,
                        tr_rmse  + tr_std, alpha=0.12, color=accent)
    ax_err.plot(train_sizes, tst_rmse, 's--', color='tomato',
                lw=2, label='Cross-val RMSE (Test)')
    ax_err.fill_between(train_sizes,
                        tst_rmse - tst_std,
                        tst_rmse + tst_std, alpha=0.12, color='tomato')
    ax_err.set_xlabel("Training Set Size", fontsize=12)
    ax_err.set_ylabel("RMSE (Error)", fontsize=12)
    ax_err.set_title(f"{name} – Error Rate Graph (Learning Curve)",
                     fontsize=13, fontweight='bold')
    ax_err.legend(fontsize=10)
    ax_err.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname_err = f"step7_{short}_error_rate.png"
    plt.savefig(fname_err, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Error rate graph saved → {fname_err}")

# ─── Takeaways ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("HYBRID MODELS TAKEAWAYS")
print("="*60)
print("• Hybrid 1 (GB + SVR): GB handles tree-structured patterns; SVR adds smooth boundary generalisation")
print("• Hybrid 2 (XGB + MLP): XGB captures feature interactions; MLP learns latent representations")
print("• StackingRegressor (cv=5) prevents data leakage during meta-learner training")
print("• Ridge meta-learner provides a regularised blend of base model outputs")
print("• Graph 3 (Base-Model Comparison): points should cluster near center if both bases agree")
print("• Graph 5 (CDF): steeper curve = errors tightly clustered near zero = better model")
print("• ROC curves (OvR) show per-class discriminability; AUC near 1 = strong separation")
print("• Error rate graphs: gap between train & test RMSE shows overfitting potential")
print("• Compare Test RMSE and R² against Step 7 (RF & GB alone) to measure hybrid gain")

