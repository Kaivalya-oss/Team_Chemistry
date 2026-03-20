# ============================================================
# STEP 7: RANDOM FOREST & GRADIENT BOOSTING
# Project: Team Chemistry in Football
# Goal: Compare tree-based models + 4 visualisations + confusion matrices
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, r2_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

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

# ─── 2. Define models ───────────────────────────────────────────────
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
}

# ─── 3. Train & evaluate each model ─────────────────────────────────
results = []
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    predictions[name] = y_pred_test

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    train_r2   = r2_score(y_train, y_pred_train)
    test_r2    = r2_score(y_test,  y_pred_test)

    results.append({
        'Model':      name,
        'Train RMSE': train_rmse,
        'Test RMSE':  test_rmse,
        'Train R²':   train_r2,
        'Test R²':    test_r2
    })

    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=features)
        print(f"\n{name} Feature Importance:")
        print(imp.sort_values(ascending=False).round(4))

# ─── 4. Summary table ───────────────────────────────────────────────
results_df = pd.DataFrame(results).round(3)
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df)

# ════════════════════════════════════════════════════════════════════
#  4 GRAPHS PER MODEL  (Random Forest first, then Gradient Boosting)
# ════════════════════════════════════════════════════════════════════

for idx, (name, model) in enumerate(models.items()):
    y_pred_test = predictions[name]
    residuals   = y_test - y_pred_test
    importances = pd.Series(model.feature_importances_, index=features)
    y_pred_cls  = pd.cut(y_pred_test, bins=bins, labels=labels)

    short = "RF" if "Forest" in name else "GB"         # short label for filenames

    # ── 2×2 figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{name} – Model Visualisations",
                 fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Graph 1: Actual vs Predicted scatter ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred_test, alpha=0.55,
                edgecolors='darkorange' if "Forest" in name else 'mediumseagreen',
                facecolors='moccasin'   if "Forest" in name else 'palegreen',
                s=60, label='Test points')
    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax1.plot(lims, lims, 'r--', lw=1.8, label='Perfect fit')
    ax1.set_xlabel("Actual Chemistry Index")
    ax1.set_ylabel("Predicted Chemistry Index")
    ax1.set_title("1. Actual vs Predicted (Test Set)")
    ax1.legend(fontsize=8)

    # Graph 2: Residual distribution ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    color2 = 'darkorange' if "Forest" in name else 'mediumseagreen'
    sns.histplot(residuals, kde=True, color=color2, ax=ax2, bins=20)
    ax2.axvline(0, color='red', linestyle='--', lw=1.5)
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("2. Residual Distribution (Test Set)")

    # Graph 3: Feature Importance heatmap ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    imp_sorted = importances.sort_values(ascending=False)
    imp_pivot  = imp_sorted.to_frame(name='Importance').T
    sns.heatmap(imp_pivot, annot=True, fmt=".4f",
                cmap="YlOrRd", linewidths=0.5, ax=ax3,
                cbar_kws={"shrink": 0.7})
    ax3.set_title("3. Feature Importance Heatmap")
    ax3.set_ylabel("")
    ax3.set_xlabel("")

    # Graph 4: Predicted vs Residual scatter (heteroscedasticity check)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(y_pred_test, residuals, alpha=0.5,
                edgecolors='steelblue', facecolors='lightskyblue', s=50)
    ax4.axhline(0, color='red', linestyle='--', lw=1.5)
    ax4.set_xlabel("Predicted Chemistry Index")
    ax4.set_ylabel("Residual")
    ax4.set_title("4. Predicted vs Residual Plot")

    plt.tight_layout()
    fname_vis = f"step7_{short}_visualisations.png"
    plt.savefig(fname_vis, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n{name} figure saved → {fname_vis}")

    # ── Confusion Matrix ─────────────────────────────────────────────
    cm = confusion_matrix(y_test_cls, y_pred_cls, labels=labels)

    fig2, ax = plt.subplots(figsize=(6, 5))
    cmap2 = 'Oranges' if "Forest" in name else 'Greens'
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap2,
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

    # ── ROC Curve (One-vs-Rest, multi-class) ─────────────────────────
    # Binarize the true labels for OvR ROC computation
    y_test_bin = label_binarize(y_test_cls, classes=labels)   # shape (n, 3)

    # For each class, use the raw predicted score as a proxy:
    #   Low    → higher score means LESS likely Low  → use  -y_pred_test
    #   Medium → closeness to mid-range (50) → use -|y_pred_test - 50|
    #   High   → higher score means MORE likely High → use  +y_pred_test
    class_scores = [
        -y_pred_test,                   # score for 'Low'
        -np.abs(y_pred_test - 50),      # score for 'Medium'
         y_pred_test,                   # score for 'High'
    ]

    roc_color = ['#e67e22', '#8e44ad', '#2980b9']  # one colour per class

    fig3, ax_roc = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], class_scores[i])
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=roc_color[i], lw=2,
                    label=f"{cls} (AUC = {roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.4, label='Random classifier')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title(f"{name} – ROC Curve (One-vs-Rest)\n"
                     f"(Low <33 | Medium 33–66 | High >66)",
                     fontsize=12, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fname_roc = f"step7_{short}_roc_curve.png"
    plt.savefig(fname_roc, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved → {fname_roc}")

    # ── Error Rate Graph (RMSE vs n_estimators) ──────────────────────
    fig4, ax_err = plt.subplots(figsize=(8, 5))

    if "Forest" in name:
        # Random Forest: incrementally add trees using warm_start
        n_range      = list(range(10, 110, 10))
        train_errors = []
        test_errors  = []
        rf_warm = RandomForestRegressor(
            n_estimators=1, max_depth=8,
            random_state=42, n_jobs=-1, warm_start=True
        )
        for n in n_range:
            rf_warm.set_params(n_estimators=n)
            rf_warm.fit(X_train, y_train)
            train_errors.append(np.sqrt(mean_squared_error(y_train, rf_warm.predict(X_train))))
            test_errors.append(np.sqrt(mean_squared_error(y_test,  rf_warm.predict(X_test))))

        ax_err.plot(n_range, train_errors, 'o-', color='darkorange',
                    lw=2, label='Train RMSE')
        ax_err.plot(n_range, test_errors,  's--', color='tomato',
                    lw=2, label='Test RMSE')
        ax_err.set_xlabel("Number of Trees (n_estimators)", fontsize=12)

    else:
        # Gradient Boosting: use staged_predict → RMSE at each boosting stage
        train_errors = [np.sqrt(mean_squared_error(y_train, yp))
                        for yp in model.staged_predict(X_train)]
        test_errors  = [np.sqrt(mean_squared_error(y_test,  yp))
                        for yp in model.staged_predict(X_test)]
        n_range = list(range(1, len(train_errors) + 1))

        ax_err.plot(n_range, train_errors, '-',  color='mediumseagreen',
                    lw=1.5, label='Train RMSE')
        ax_err.plot(n_range, test_errors,  '--', color='tomato',
                    lw=1.5, label='Test RMSE')
        ax_err.set_xlabel("Boosting Round (n_estimators)", fontsize=12)

    ax_err.set_ylabel("RMSE (Error)", fontsize=12)
    ax_err.set_title(f"{name} – Error Rate Graph", fontsize=13, fontweight='bold')
    ax_err.legend(fontsize=10)
    ax_err.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname_err = f"step7_{short}_error_rate.png"
    plt.savefig(fname_err, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Error rate graph saved → {fname_err}")

# ─── 5. Takeaways ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7 TAKEAWAYS")
print("="*60)
print("• If Test R² > Linear Regression → trees capture better patterns")
print("• If Test RMSE much lower → non-linear effects or interactions exist")
print("• Feature importance heatmap (graph 3) shows which proxy drives predictions")
print("• Predicted vs Residual plot (graph 4) checks for heteroscedasticity")
print("• Confusion matrices bin predictions: Low / Medium / High chemistry")
print("• ROC curves (OvR) show class-level discriminability; AUC near 1 = strong separation")
print("• Error rate graphs: decreasing test RMSE = model improves; flattening = convergence")
print("Next steps: Hyperparameter tuning (Step 8), cross-validation (Step 10), interpretation (Step 11)")