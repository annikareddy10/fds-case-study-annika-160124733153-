# Fundamentals of Data Science
# Case Study 1: Online Retail
# Student: Annika Reddy
# Roll Number: 160124733153

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, AgglomerativeClustering,
    SpectralClustering, DBSCAN
)
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    pairwise_distances
)
from IPython.display import display

warnings.filterwarnings("ignore")

RANDOM_STATE = 7
OUTPUT_DIR = "outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# 1. Load dataset
data_path = Path("Online retail_new(1).csv")

if not data_path.exists():
    from google.colab import files
    files.upload()

retail = pd.read_csv(data_path, header=None)

print("Dataset loaded:", data_path.name)
print("Dataset shape:", retail.shape)
display(retail.head())

# 2. Clean baskets and create basket-product matrix
baskets = []

for _, row in retail.iterrows():
    items = [
        str(v).strip().lower()
        for v in row.tolist()
        if pd.notna(v) and str(v).strip()
    ]
    baskets.append(sorted(set(items)))

basket_sizes = np.array([len(b) for b in baskets])
products = sorted({
    item for basket in baskets for item in basket
})

basket_matrix = np.zeros(
    (len(baskets), len(products)), dtype=np.uint8
)
product_to_col = {
    product: i for i, product in enumerate(products)
}

for i, basket in enumerate(baskets):
    for item in basket:
        basket_matrix[i, product_to_col[item]] = 1

basket_df = pd.DataFrame(
    basket_matrix, columns=products
)

product_counts = basket_df.sum().sort_values(
    ascending=False
)
product_percent = 100 * product_counts / len(basket_df)

summary = pd.DataFrame({
    "Metric": [
        "Baskets", "Distinct products",
        "Basket-product entries", "Mean basket size",
        "Std basket size", "Median basket size",
        "Q1 basket size", "Q3 basket size",
        "Single-item baskets (%)",
        "Baskets with 8+ items (%)"
    ],
    "Value": [
        len(baskets), len(products),
        int(basket_matrix.sum()),
        basket_sizes.mean(),
        basket_sizes.std(ddof=1),
        np.median(basket_sizes),
        np.percentile(basket_sizes, 25),
        np.percentile(basket_sizes, 75),
        100 * np.mean(basket_sizes == 1),
        100 * np.mean(basket_sizes >= 8)
    ]
})

display(summary.round(3))
summary.to_csv(
    f"{OUTPUT_DIR}/retail_summary.csv", index=False
)

freq = pd.DataFrame({
    "Product": product_counts.index,
    "Basket count": product_counts.values,
    "Basket share (%)": product_percent.values
})

display(freq.head(15).round(3))
freq.to_csv(
    f"{OUTPUT_DIR}/retail_product_frequencies.csv",
    index=False
)

pd.DataFrame({
    "Basket size": basket_sizes
}).to_csv(
    f"{OUTPUT_DIR}/retail_basket_sizes.csv",
    index=False
)

# Top products graph
freq.head(15).sort_values(
    "Basket share (%)"
).plot(
    x="Product",
    y="Basket share (%)",
    kind="barh",
    figsize=(9, 6),
    legend=False,
    title="Top 15 Products by Basket Share"
)

plt.xlabel("Baskets containing product (%)")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/retail_top_products.png", dpi=200
)
plt.show()

# Basket-size distribution graph
pd.Series(basket_sizes).value_counts().sort_index().plot(
    kind="bar",
    figsize=(9, 5),
    title="Basket-size Distribution"
)

plt.xlabel("Number of products in basket")
plt.ylabel("Number of baskets")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/retail_basket_size_distribution.png",
    dpi=200
)
plt.show()

# 3. Pairwise association rules
n = len(basket_df)
min_support_count = int(np.ceil(0.01 * n))

eligible = product_counts[
    product_counts >= min_support_count
].index.tolist()

assoc = basket_df[eligible]
rows = []

for a in eligible:
    a_count = int(assoc[a].sum())

    for b in eligible:
        if a == b:
            continue

        b_count = int(assoc[b].sum())
        joint = int(
            ((assoc[a] == 1) & (assoc[b] == 1)).sum()
        )

        if joint >= 75 and a_count and b_count:
            confidence = joint / a_count
            support = joint / n
            lift = confidence / (b_count / n)

            rows.append((
                a, b, joint, support, confidence, lift
            ))

rules = pd.DataFrame(
    rows,
    columns=[
        "If basket has", "Then also", "Joint baskets",
        "Support", "Confidence", "Lift"
    ]
)

rules = rules.sort_values(
    ["Lift", "Confidence"], ascending=False
)

display(rules.head(20).round(4))
rules.to_csv(
    f"{OUTPUT_DIR}/retail_association_rules.csv",
    index=False
)

# 4. PCA and clustering
rng = np.random.RandomState(RANDOM_STATE)
sample_n = min(2000, len(basket_df))

idx = rng.choice(
    len(basket_df), sample_n, replace=False
)
X = basket_df.iloc[idx].to_numpy()

pca = PCA(random_state=RANDOM_STATE)
Xp = pca.fit_transform(X)

pca_table = pd.DataFrame({
    "Component": np.arange(
        1, len(pca.explained_variance_ratio_) + 1
    ),
    "Variance explained": pca.explained_variance_ratio_,
    "Cumulative variance": np.cumsum(
        pca.explained_variance_ratio_
    )
})

display(pca_table.head(10).round(4))
pca_table.to_csv(
    f"{OUTPUT_DIR}/retail_pca_variance.csv",
    index=False
)

print(
    "First two components variance (%):",
    (pca.explained_variance_ratio_[:2] * 100).round(2)
)
print(
    "Cumulative variance first 10 components (%):",
    round(pca.explained_variance_ratio_[:10].sum() * 100, 2)
)

plt.figure(figsize=(8, 5))
plt.bar(
    range(1, min(10, len(pca.explained_variance_ratio_)) + 1),
    pca.explained_variance_ratio_[:10] * 100
)
plt.xlabel("Principal component")
plt.ylabel("Variance explained (%)")
plt.title("PCA Scree Plot")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/retail_pca_scree.png", dpi=200
)
plt.show()

# K-means validation
km_results = []

for k in range(2, 9):
    labels = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=RANDOM_STATE
    ).fit_predict(X)

    km_results.append((
        k,
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels)
    ))

km_df = pd.DataFrame(
    km_results,
    columns=["k", "Silhouette", "Davies-Bouldin"]
)

display(km_df.round(4))
km_df.to_csv(
    f"{OUTPUT_DIR}/retail_kmeans_validation.csv",
    index=False
)

km_df.set_index("k")[
    ["Silhouette", "Davies-Bouldin"]
].plot(
    marker="o",
    subplots=True,
    figsize=(8, 6),
    title="K-means Validation Scores"
)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/retail_kmeans_validation.png",
    dpi=200
)
plt.show()

# Compare clustering methods
cluster_models = {
    "K-means (k=5)": KMeans(
        n_clusters=5, n_init=10,
        random_state=RANDOM_STATE
    ),
    "Hierarchical (complete, k=5)": (
        AgglomerativeClustering(
            n_clusters=5, linkage="complete"
        )
    ),
    "Spectral (k=5)": SpectralClustering(
        n_clusters=5,
        affinity="nearest_neighbors",
        n_neighbors=min(15, len(X) - 1),
        random_state=RANDOM_STATE
    )
}

cluster_rows = []

for name, model in cluster_models.items():
    labels = model.fit_predict(X)

    cluster_rows.append((
        name,
        len(set(labels)),
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels)
    ))

D = pairwise_distances(X, metric="jaccard")

db_labels = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric="precomputed"
).fit_predict(D)

mask = db_labels != -1
db_clusters = len(set(db_labels[mask]))
db_silhouette = np.nan

if mask.sum() > 1 and db_clusters > 1:
    db_silhouette = silhouette_score(
        X[mask], db_labels[mask]
    )

cluster_rows.append((
    "DBSCAN", db_clusters, db_silhouette, np.nan
))

cluster_metrics = pd.DataFrame(
    cluster_rows,
    columns=[
        "Method", "Clusters excluding noise",
        "Silhouette", "Davies-Bouldin"
    ]
)

display(cluster_metrics.round(4))
cluster_metrics.to_csv(
    f"{OUTPUT_DIR}/retail_clustering_metrics.csv",
    index=False
)

print(
    "DBSCAN noise points:", int((~mask).sum()),
    "out of", len(X)
)

# PCA projection graph
plt.figure(figsize=(8, 6))
plt.scatter(Xp[:, 0], Xp[:, 1], s=12, alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Retail Baskets: PCA Projection")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/retail_pca_projection.png", dpi=200
)
plt.show()

# 5. List saved tables and graphs
output_files = sorted(Path(OUTPUT_DIR).glob("*"))

output_inventory = pd.DataFrame({
    "File": [p.name for p in output_files],
    "Type": [
        "Figure (PNG)" if p.suffix.lower() == ".png"
        else "Table (CSV)" if p.suffix.lower() == ".csv"
        else p.suffix
        for p in output_files
    ],
    "Size (bytes)": [
        p.stat().st_size for p in output_files
    ]
})

display(output_inventory)
