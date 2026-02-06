# ============================================================
# STEP 1: DATA UNDERSTANDING & FEATURE SELECTION
# Project: Team Chemistry in Football
# Dataset: fifa_eda_stats.xlsx
# ============================================================
 
# -----------------------------
# Import required libraries
# -----------------------------
 
import pandas as pd                  # Data loading and manipulation
import numpy as np                   # Numerical computations
import matplotlib.pyplot as plt      # Visualization
import seaborn as sns                # type: ignore # Statistical visualization
import warnings                      # Warning control
 
# Suppress warnings for clean output
warnings.filterwarnings("ignore")
 
# -----------------------------
# Load the dataset
# -----------------------------
 
# Read the Excel file (same folder as notebook)
df = pd.read_excel("fifa_eda_stats.xlsx")
 
# -----------------------------
# Initial data inspection
# -----------------------------
 
df.head()        # View first 5 rows
df.shape         # Dataset dimensions
df.info()        # Column names and data types
df.describe()    # Summary statistics
 
# -----------------------------
# Missing value check
# -----------------------------
 
df.isnull().sum()
 
# -----------------------------
# Separate numerical & categorical features
# -----------------------------
 
# Identify numerical columns
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
 
# Identify categorical columns
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
 
# -----------------------------
# Remove irrelevant identifier columns (if present)
# -----------------------------
 
irrelevant_columns = [
    "player_id",
    "sofifa_id",
    "player_url",
    "photo_url"
]
 
# Drop columns safely if they exist
df.drop(
    columns=[col for col in irrelevant_columns if col in df.columns],
    inplace=True
)
 
# Update numerical feature list after dropping columns
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
 
# -----------------------------
# Variance Threshold Filtering
# -----------------------------
 
# Calculate variance for numerical features
variance = df[numerical_features].var()
 
# Set minimum variance threshold
variance_threshold = 0.01
 
# Select features above variance threshold
selected_variance_features = variance[
    variance > variance_threshold
].index.tolist()
 
# Keep filtered numerical + all categorical features
df = df[selected_variance_features + categorical_features]
 
# -----------------------------
# Correlation Heatmap (EDA)
# -----------------------------
 
# Compute correlation matrix
correlation_matrix = df[selected_variance_features].corr()
 
# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
 
# ============================================================
# VIF CALCULATION (ERROR-FREE)
# ============================================================
 
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore
 
# -----------------------------
# Prepare data for VIF
# -----------------------------
 
# Copy numerical features only
df_vif = df[selected_variance_features].copy()
 
# Replace infinite values with NaN
df_vif.replace([np.inf, -np.inf], np.nan, inplace=True)
 
# Fill missing values with column mean
df_vif.fillna(df_vif.mean(), inplace=True)
 
# -----------------------------
# Compute VIF values
# -----------------------------
 
vif_data = pd.DataFrame()
vif_data["Feature"] = df_vif.columns
 
vif_data["VIF"] = [
    variance_inflation_factor(df_vif.values, i)
    for i in range(df_vif.shape[1])
]
 
# Display VIF table
vif_data.sort_values(by="VIF", ascending=False)
 
# -----------------------------
# Select final features using VIF
# -----------------------------
 
# Define VIF threshold
vif_threshold = 10
 
# Retain features with acceptable multicollinearity
final_numerical_features = vif_data[
    vif_data["VIF"] < vif_threshold
]["Feature"].tolist()
 
# -----------------------------
# Create final dataset
# -----------------------------
 
# Combine numerical + categorical features
df_final = df[final_numerical_features + categorical_features]
 
# -----------------------------
# Final output check
# -----------------------------
 
df_final.head()      # Preview final dataset
df_final.shape       # Final dataset size
df_final.columns     # Final selected features