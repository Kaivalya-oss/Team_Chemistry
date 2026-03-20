# ============================================================
# STEP 6: LINEAR REGRESSION MODELING
# Project: Team Chemistry in Football
# Goal: Fit linear model + evaluate + 4 visualisations + confusion matrix
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
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

print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)

# ─── 2. Train Linear Regression ─────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ─── 3. Make predictions ────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# ─── 4. Performance metrics ─────────────────────────────────────────
def print_metrics(y_true, y_pred, dataset_name):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{dataset_name} Performance:")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.3f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test,  y_pred_test,  "Test")

# ─── 5. Coefficients (feature importance) ───────────────────────────
coef_df = pd.DataFrame({
    'Feature':     X_train.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients (higher = stronger positive impact):")
print(coef_df.round(3))
print(f"\nIntercept: {model.intercept_:.2f}")

# ─── Helper: bin continuous predictions into 3 classes ──────────────
#   Low (<33), Medium (33–66), High (>66)
bins   = [-np.inf, 33, 66, np.inf]
labels = ['Low', 'Medium', 'High']

y_test_cls  = pd.cut(y_test,       bins=bins, labels=labels)
y_pred_cls  = pd.cut(y_pred_test,  bins=bins, labels=labels)

# ────────────────────────────────────────────────────────────────────
#  4 GRAPHS  –  arranged in a 2 × 2 grid
# ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Linear Regression – Model Visualisations", fontsize=16, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Graph 1: Actual vs Predicted scatter ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred_test, alpha=0.55, edgecolors='steelblue',
            facecolors='lightskyblue', s=60, label='Test points')
lims = [min(y_test.min(), y_pred_test.min()),
        max(y_test.max(), y_pred_test.max())]
ax1.plot(lims, lims, 'r--', lw=1.8, label='Perfect fit')
ax1.set_xlabel("Actual Chemistry Index")
ax1.set_ylabel("Predicted Chemistry Index")
ax1.set_title("1. Actual vs Predicted (Test Set)")
ax1.legend(fontsize=8)

# ── Graph 2: Residual distribution (histogram + KDE) ────────────────
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test - y_pred_test
sns.histplot(residuals, kde=True, color='steelblue', ax=ax2, bins=20)
ax2.axvline(0, color='red', linestyle='--', lw=1.5)
ax2.set_xlabel("Residual (Actual − Predicted)")
ax2.set_ylabel("Count")
ax2.set_title("2. Residual Distribution (Test Set)")

# ── Graph 3: Feature-coefficient heatmap ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
coef_pivot = coef_df.set_index('Feature')[['Coefficient']].T
sns.heatmap(coef_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax3,
            cbar_kws={"shrink": 0.7})
ax3.set_title("3. Feature Coefficient Heatmap")
ax3.set_ylabel("")
ax3.set_xlabel("")

# ── Graph 4: Feature Correlation Matrix (train data) ────────────────
ax4 = fig.add_subplot(gs[1, 1])
corr = pd.concat([X_train, y_train.rename('Chemistry_Index_100')], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax4,
            cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
ax4.set_title("4. Feature Correlation Matrix")
ax4.tick_params(axis='x', labelrotation=30, labelsize=7)
ax4.tick_params(axis='y', labelrotation=0,  labelsize=7)

plt.tight_layout()
plt.savefig("step6_visualisations.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved → step6_visualisations.png")

# ────────────────────────────────────────────────────────────────────
#  CONFUSION MATRIX  (binned: Low / Medium / High)
# ────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test_cls, y_pred_cls, labels=labels)

fig2, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_xlabel("Predicted Class",  fontsize=11)
ax.set_ylabel("Actual Class",     fontsize=11)
ax.set_title("Linear Regression – Confusion Matrix\n(Low <33 | Medium 33–66 | High >66)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("step6_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved → step6_confusion_matrix.png")

print("\nClassification Report (binned classes):")
print(classification_report(y_test_cls, y_pred_cls, labels=labels, zero_division=0))

# ────────────────────────────────────────────────────────────────────
#  ERROR RATE GRAPH – Learning Curve (Train vs Test RMSE)
# ────────────────────────────────────────────────────────────────────
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(),
    pd.concat([X_train, X_test]),
    pd.concat([y_train, y_test]),
    cv=5,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_rmse = np.sqrt(-train_scores.mean(axis=1))
test_rmse  = np.sqrt(-test_scores.mean(axis=1))
train_std  = np.sqrt(train_scores.std(axis=1))
test_std   = np.sqrt(test_scores.std(axis=1))

fig3, ax_lc = plt.subplots(figsize=(8, 5))
ax_lc.plot(train_sizes, train_rmse, 'o-', color='steelblue',
           lw=2, label='Train RMSE')
ax_lc.fill_between(train_sizes,
                   train_rmse - train_std,
                   train_rmse + train_std,
                   alpha=0.15, color='steelblue')
ax_lc.plot(train_sizes, test_rmse, 's--', color='tomato',
           lw=2, label='Cross-val RMSE (Test)')
ax_lc.fill_between(train_sizes,
                   test_rmse - test_std,
                   test_rmse + test_std,
                   alpha=0.15, color='tomato')
ax_lc.set_xlabel("Training Set Size", fontsize=12)
ax_lc.set_ylabel("RMSE (Error)",      fontsize=12)
ax_lc.set_title("Linear Regression – Learning Curve (Error Rate)",
                fontsize=13, fontweight='bold')
ax_lc.legend(fontsize=10)
ax_lc.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("step6_error_rate.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nError rate graph saved → step6_error_rate.png")

# ─── Quick interpretation ────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 TAKEAWAYS")
print("="*60)
print("• R² on test set shows how much variance the 4 features explain")
print("• Coefficients reveal which proxy matters most (heatmap, graph 3)")
print("• Residual histogram (graph 2) reveals bias / heteroscedasticity")
print("• Correlation matrix (graph 4) shows multicollinearity among features")
print("• Confusion matrix bins predictions: Low / Medium / High chemistry")
print("• Learning curve (error rate): gap between train & test RMSE shows overfitting/underfitting")
print("Next: Step 7 – try Random Forest / Gradient Boosting for comparison")