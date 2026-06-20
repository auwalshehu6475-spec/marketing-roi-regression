import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load and Clean the Dataset
df = pd.read_csv("b9de2fb4-306d-42f1-b589-800f4d6e3698.csv")

# Drop only rows where the clustering features are missing, not the whole row
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
df_clean = df.dropna(subset=features).reset_index(drop=True)

# Extract numeric features for clustering
X = df_clean[features]

# 2. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine Optimal Clusters (Elbow & Silhouette Methods)
sse = []
silhouette_scores = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotting the Evaluation Metrics
fig, ax1 = plt.subplots(figsize=(10, 5))

# Elbow Curve
color = "tab:blue"
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("SSE (Inertia)", color=color)
ax1.plot(k_range, sse, marker="o", color=color, linewidth=2)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_title("Elbow Method and Silhouette Scores for Optimal k")

# Silhouette Score
ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Silhouette Score", color=color)
ax2.plot(k_range, silhouette_scores, marker="s", color=color, linewidth=2)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.show()

# Print metrics for confirmation
for k, s, sil in zip(k_range, sse, silhouette_scores):
    print(f"k={k} -> SSE: {s:.2f} | Silhouette Score: {sil:.3f}")

# 4. Fit Final K-means Model (Choosing k=3 based on species and validation metrics)
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clean["Cluster"] = final_kmeans.fit_predict(X_scaled)

# 5. Visualize Cluster Assignments
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_clean,
    x="flipper_length_mm",
    y="body_mass_g",
    hue="Cluster",
    style="species",
    palette="Set1",
    s=100,
    alpha=0.8,
)
plt.title("Penguin Clusters: Flipper Length vs. Body Mass")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 0.85, 1])   # make room for the legend
plt.show()

# 6. Biological Alignment Cross-tabulation
print("\n--- Alignment with Known Species ---")
print(pd.crosstab(df_clean["species"], df_clean["Cluster"]))

print("\n--- Alignment with Sex (Size/Sexual Dimorphism) ---")
# Note: rows where 'sex' is NaN are automatically excluded from the crosstab
print(pd.crosstab(df_clean["sex"], df_clean["Cluster"]))

# 7. Cluster Centroids (Unscaled Mean Values)
print("\n--- Profiling Cluster Centroids ---")
print(df_clean.groupby("Cluster")[features].mean())