import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from your ML_values.txt
data = {
    'Model': [
        'Linear Regression', 
        'Random Forest (untuned)', 
        'Gradient Boosting (untuned)', 
        'Random Forest (tuned)', 
        'Gradient Boosting (tuned)'
    ],
    'Test R²':   [1.0000, 0.9290, 0.9730, 0.9402, 0.9832],
    'Test RMSE': [0.0000, 2.5880, 1.6070, 2.3798, 1.2613]
}

df_comp = pd.DataFrame(data)

# Sort by Test R² descending (best first)
df_comp = df_comp.sort_values('Test R²', ascending=False).reset_index(drop=True)

print("\n" + "="*60)
print("FINAL MODEL COMPARISON – TEST SET PERFORMANCE")
print("="*60)
print(df_comp.round(4))

# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=df_comp, x='Test R²', y='Model', ax=axes[0], palette='viridis')
axes[0].set_title('Test R² (higher = better)')
axes[0].set_xlim(0.85, 1.01)

sns.barplot(data=df_comp, x='Test RMSE', y='Model', ax=axes[1], palette='viridis')
axes[1].set_title('Test RMSE (lower = better)')

plt.suptitle("Model Performance Comparison (Test Set)", fontsize=14)
plt.tight_layout()
plt.show()