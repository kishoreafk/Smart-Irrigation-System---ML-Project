import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('TARP.csv/TARP.csv')
df.columns = df.columns.str.strip()

# Set style
sns.set_style("whitegrid")
plt.rcParams['agg.path.chunksize'] = 10000

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nStatistical Summary:\n{df.describe()}")
print(f"\nStatus Distribution:\n{df['Status'].value_counts()}")

# 1. Correlation Heatmap
plt.figure(figsize=(14, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('01_correlation_heatmap.png', dpi=300)
plt.close()

# 2. Bar Charts - Status Count
plt.figure(figsize=(8, 6))
df['Status'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Status Distribution (ON vs OFF)')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('02_bar_status_count.png', dpi=300)
plt.close()

# 3. Bar Charts - Average values by Status
features = ['Soil Moisture', 'Temperature', 'rainfall', 'N', 'P', 'K']
for col in features:
    plt.figure(figsize=(8, 6))
    df.groupby('Status')[col].mean().plot(kind='bar', color=['green', 'red'])
    plt.title(f'Average {col} by Status')
    plt.xlabel('Status')
    plt.ylabel(f'Average {col}')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'03_bar_avg_{col.replace(" ", "_")}.png', dpi=300)
    plt.close()

# 4. Box Plots - Key Features by Status
features = ['Soil Moisture', 'Temperature', 'Soil Humidity', 'rainfall', 'ph', 'N', 'P', 'K']
for col in features:
    plt.figure(figsize=(8, 6))
    df.boxplot(column=col, by='Status')
    plt.title(f'{col} Distribution by Status')
    plt.xlabel('Status')
    plt.ylabel(col)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'04_box_{col.replace(" ", "_")}.png', dpi=300)
    plt.close()

# 5. Pie Charts - Status Distribution
plt.figure(figsize=(8, 8))
df['Status'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title('Status Distribution (ON vs OFF)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('05_pie_status.png', dpi=300)
plt.close()

# 6. Pie Charts - NPK Ranges
for nutrient in ['N', 'P', 'K']:
    plt.figure(figsize=(8, 8))
    bins = [0, 50, 100, 150]
    labels = ['Low', 'Medium', 'High']
    df[f'{nutrient}_range'] = pd.cut(df[nutrient], bins=bins, labels=labels)
    df[f'{nutrient}_range'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f'{nutrient} Level Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'06_pie_{nutrient}_levels.png', dpi=300)
    plt.close()

# 7. Scatter Plots - Key Relationships
scatter_pairs = [
    ('Soil Moisture', 'Temperature'),
    ('Air temperature (C)', 'Air humidity (%)'),
    ('N', 'P'),
    ('P', 'K'),
    ('rainfall', 'Soil Moisture'),
    ('ph', 'Soil Humidity')
]
for x, y in scatter_pairs:
    plt.figure(figsize=(10, 6))
    for status in df['Status'].unique():
        mask = df['Status'] == status
        plt.scatter(df[mask][x], df[mask][y], label=status, alpha=0.6, s=30)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'07_scatter_{x.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}_vs_{y.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png', dpi=300)
    plt.close()

# 8. Line Plots - Trends over Time (using index as proxy)
features = ['Soil Moisture', 'Temperature', 'rainfall', 'Air temperature (C)']
for col in features:
    plt.figure(figsize=(12, 6))
    # Sample every 10th point to reduce data
    sample_df = df[::10].reset_index(drop=True)
    plt.plot(sample_df.index, sample_df[col], marker='o', markersize=3, linewidth=1)
    plt.title(f'{col} Trend (Sampled Data)')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'08_line_{col.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300)
    plt.close()

# 9. Line Plots - NPK Trends
plt.figure(figsize=(12, 6))
sample_df = df[::10].reset_index(drop=True)
plt.plot(sample_df.index, sample_df['N'], label='N', marker='o', markersize=3, linewidth=1)
plt.plot(sample_df.index, sample_df['P'], label='P', marker='s', markersize=3, linewidth=1)
plt.plot(sample_df.index, sample_df['K'], label='K', marker='^', markersize=3, linewidth=1)
plt.title('NPK Nutrients Trend (Sampled Data)')
plt.xlabel('Sample Index')
plt.ylabel('Nutrient Level')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('09_line_NPK_trend.png', dpi=300)
plt.close()

# 10. Histograms - Distribution Analysis
features = ['Soil Moisture', 'Temperature', 'rainfall', 'ph', 'N', 'P', 'K']
for col in features:
    plt.figure(figsize=(10, 6))
    plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'10_hist_{col.replace(" ", "_")}.png', dpi=300)
    plt.close()

# 11. Scatter Plot Matrix for NPK
plt.figure(figsize=(10, 10))
pd.plotting.scatter_matrix(df[['N', 'P', 'K']], alpha=0.6, figsize=(10, 10), diagonal='hist')
plt.suptitle('NPK Scatter Matrix', y=1.0)
plt.tight_layout()
plt.savefig('11_scatter_matrix_NPK.png', dpi=300)
plt.close()

# 12. Box Plot - Overall Distribution
plt.figure(figsize=(14, 6))
features_to_plot = ['Soil Moisture', 'Temperature', 'rainfall', 'ph']
df[features_to_plot].boxplot()
plt.title('Overall Distribution of Key Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('12_box_overall.png', dpi=300)
plt.close()

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nGenerated plots:")
print("- Correlation heatmap")
print("- Bar charts (status counts and averages)")
print("- Box plots (distributions by status)")
print("- Pie charts (status and NPK levels)")
print("- Scatter plots (relationships)")
print("- Line plots (trends)")
print("- Histograms (distributions)")
print("- Scatter matrix (NPK)")
print("\nAll plots saved as PNG files!")
