import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Select relevant features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Elbow Method
# --------------------------
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.savefig("elbow_method.png")
plt.close()

# --------------------------
# KMeans with optimal K
# --------------------------
optimal_k = 5  # from elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# --------------------------
# Cluster Visualization
# --------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", 
                hue="Cluster", data=df, palette="Set1", s=80)
plt.title("Customer Segmentation with KMeans")
plt.savefig("clusters.png")
plt.close()

# --------------------------
# Silhouette Score
# --------------------------
score = silhouette_score(X_scaled, df["Cluster"])
print("Silhouette Score:", score)
