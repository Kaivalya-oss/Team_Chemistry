# ============================================================
# STEP 3: IMPROVED EXPLORATORY DATA ANALYSIS (EDA)
# Project: Team Chemistry in Football
# Focus: Understand team-level patterns, especially chemistry proxies
# ============================================================

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Styling ────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
#import matplotlib.inline   # only needed in Jupyter; safe to keep

# ─── 1. Load & Quick Check ──────────────────────────────────────────
df = pd.read_csv("team_level_processed.csv")

print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values (should be 0):")
print(df.isna().sum().sum())

# ─── 2. Define feature groups for easier analysis ───────────────────
core_quality = ['Overall', 'Potential', 'Value', 'Wage', 'Age']
technical    = ['Crossing', 'Finishing', 'ShortPassing', 'LongPassing', 
                'Dribbling', 'BallControl', 'Vision', 'Composure']
physical     = ['Acceleration', 'SprintSpeed', 'Stamina', 'Strength', 
                'Aggression']
defensive    = ['Interceptions', 'Marking', 'StandingTackle', 'SlidingTackle']
chemistry_proxies = ['SquadSize', 'NationalityDiversity', 'is_LeftFoot',
                     'International Reputation', 'Skill Moves', 'Weak Foot',
                     'Attack_Work_Intensity', 'Defense_Work_Intensity']

all_numeric = core_quality + technical + physical + defensive + chemistry_proxies

# ─── 3. Summary Statistics Table ────────────────────────────────────
print("\nKey Summary Statistics (rounded):")
print(df[all_numeric].describe().round(2))

# ─── 4. Distribution Plots – Selected Important Ones ────────────────
important_vars = ['Overall', 'Potential', 'Value', 'Age', 
                  'NationalityDiversity', 'is_LeftFoot',
                  'Attack_Work_Intensity', 'Defense_Work_Intensity']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, var in enumerate(important_vars):
    sns.histplot(data=df, x=var, kde=True, ax=axes[i], color='cornflowerblue')
    axes[i].set_title(f'Distribution of {var}', fontsize=11)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Count')

plt.suptitle("Key Feature Distributions (z-scores except SquadSize)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ─── 5. Correlation Heatmap – Focused version ───────────────────────
plt.figure(figsize=(13, 11))
corr_matrix = df[all_numeric].corr()
sns.heatmap(corr_matrix, 
            cmap='RdBu_r', center=0, 
            annot=False, fmt='.2f', 
            linewidths=0.4, 
            cbar_kws={'label': 'Correlation'})
plt.title("Correlation Matrix – Team Level Features", fontsize=14)
plt.show()

# Top correlations with Overall (our quality proxy)
print("\nTop 12 features most correlated with Overall:")
print(corr_matrix['Overall']
      .sort_values(ascending=False)
      .round(3)
      .head(12))

print("\nBottom 6 (most negative):")
print(corr_matrix['Overall']
      .sort_values(ascending=True)
      .round(3)
      .head(6))

# ─── 6. Scatter Plots – Chemistry Hypotheses ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

scatter_pairs = [
    ('Value', 'Overall', 'Market Value vs Current Quality'),
    ('NationalityDiversity', 'Overall', 'Diversity vs Quality'),
    ('is_LeftFoot', 'Overall', '% Left-Footed vs Quality'),
    ('Attack_Work_Intensity', 'Defense_Work_Intensity', 'Attack vs Defense Work Rate'),
    ('Age', 'Potential', 'Age vs Future Potential'),
    ('NationalityDiversity', 'Composure', 'Diversity vs Composure')
]

for i, (x, y, title) in enumerate(scatter_pairs):
    sns.scatterplot(data=df, x=x, y=y, alpha=0.7, ax=axes[i])
    axes[i].set_title(title, fontsize=11)

plt.suptitle("Key Relationships – Looking for Chemistry Signals", fontsize=14)
plt.tight_layout()
plt.show()

# ─── 7. Top vs Bottom Teams Comparison ──────────────────────────────
top_n = 25
bottom_n = 25

top_teams    = df.nlargest(top_n, 'Overall').copy()
bottom_teams = df.nsmallest(bottom_n, 'Overall').copy()

compare_df = pd.concat([
    top_teams.assign(Group=f'Top {top_n}'),
    bottom_teams.assign(Group=f'Bottom {bottom_n}')
])

# Boxplots – chemistry proxies
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

sns.boxplot(data=compare_df, x='Group', y='NationalityDiversity', ax=axes[0])
axes[0].set_title("Nationality Diversity")

sns.boxplot(data=compare_df, x='Group', y='is_LeftFoot', ax=axes[1])
axes[1].set_title("% Left-Footed Players")

sns.boxplot(data=compare_df, x='Group', y='International Reputation', ax=axes[2])
axes[2].set_title("Avg International Reputation")

plt.suptitle(f"Chemistry Proxies: Top {top_n} vs Bottom {bottom_n} Teams", fontsize=14)
plt.tight_layout()
plt.show()

# Quick table print
print(f"\nTop {top_n} teams – average values:")
print(top_teams[chemistry_proxies + ['Overall']].mean().round(3))

print(f"\nBottom {bottom_n} teams – average values:")
print(bottom_teams[chemistry_proxies + ['Overall']].mean().round(3))
# ─── END OF STEP 3 ─────────────────────────────────────────────────
