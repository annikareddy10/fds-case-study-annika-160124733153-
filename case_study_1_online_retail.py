
"""
Fundamentals of Data Science
Case Study 1: Online Retail

Analysis:
1. Clean transaction baskets
2. Basket-size and product-frequency analysis
3. Pairwise association rules: support, confidence, lift
4. PCA and scree analysis
5. K-means, hierarchical, spectral and DBSCAN clustering
6. Cluster evaluation and CSV exports

Input:
data/Online retail_new(1).csv

Outputs:
outputs/retail_summary.csv
outputs/retail_product_frequencies.csv
outputs/retail_basket_sizes.csv
outputs/retail_association_rules.csv
outputs/retail_kmeans_validation.csv
outputs/retail_clustering_metrics.csv
outputs/retail_pca_variance.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
)
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    pairwise_distances,
)

warnings.filterwarnings("ignore")

# Keep consistent with the current clean notebook.
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Online retail_new(1).csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data():
    """Load the headerless retail file and clean each basket."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Retail CSV not found: {DATA_FILE}\n"
            "Create a data folder and place the retail CSV inside it."
        )

    retail = pd.read_csv(
        DATA_FILE,
        header=None,
        dtype=str,
        keep_default_na=False,
    )

    baskets = []
    for _, row in retail.iterrows():
        items = [
            str(value).strip().lower()
            for value in row.tolist()
            if str(value).strip()
        ]
        # Each product is counted once per basket.
        baskets.append(sorted(set(items)))

    products = sorted({
        product
        for basket in baskets
        for product in basket
    })

    basket_matrix = np.zeros(
        (len(baskets), len(products)),
        dtype=np.uint8,
    )
    product_to_col = {
        product: index
        for index, product in enumerate(products)
    }

    for row_index, basket in enumerate(baskets):
        for product in basket:
            basket_matrix[
                row_index, product_to_col[product]
            ] = 1

    basket_df = pd.DataFrame(
        basket_matrix,
        columns=products,
    )
    basket_sizes = np.array(
        [len(basket) for basket in baskets]
    )

    return retail, baskets, basket_df, basket_sizes


def descriptive_analysis(basket_df, basket_sizes):
    """Save basket statistics, product frequencies and plots."""
    product_counts = basket_df.sum().sort_values(
        ascending=False
    )
    product_share = 100 * product_counts / len(basket_df)

    summary = pd.DataFrame({
        "Metric": [
            "Number of baskets",
            "Number of distinct products",
            "Basket-product entries",
            "Mean basket size",
            "Standard deviation of basket size",
            "Median basket size",
            "First quartile",
            "Third quartile",
            "Minimum basket size",
            "Maximum basket size",
            "Single-item baskets (%)",
            "Baskets with 8 or more items (%)",
        ],
        "Value": [
            len(basket_df),
            basket_df.shape[1],
            int(basket_df.to_numpy().sum()),
            basket_sizes.mean(),
            basket_sizes.std(ddof=1),
            np.median(basket_sizes),
            np.percentile(basket_sizes, 25),
            np.percentile(basket_sizes, 75),
            basket_sizes.min(),
            basket_sizes.max(),
            100 * np.mean(basket_sizes == 1),
            100 * np.mean(basket_sizes >= 8),
        ],
    })

    summary.to_csv(
        OUTPUT_DIR / "retail_summary.csv",
        index=False,
    )

    frequencies = pd.DataFrame({
        "Product": product_counts.index,
        "Basket count": product_counts.values,
        "Basket share (%)": product_share.values,
    })
    frequencies.to_csv(
        OUTPUT_DIR / "retail_product_frequencies.csv",
        index=False,
    )

    pd.DataFrame({
        "Basket size": basket_sizes
    }).to_csv(
        OUTPUT_DIR / "retail_basket_sizes.csv",
        index=False,
    )

    print("\nRETAIL DATASET SUMMARY")
    print(summary.to_string(index=False))

    print("\nTOP 15 PRODUCTS")
    print(frequencies.head(15).round(3).to_string(index=False))

    frequencies.head(15).sort_values(
        "Basket share (%)"
    ).plot(
        x="Product",
        y="Basket share (%)",
        kind="barh",
        figsize=(9, 6),
        legend=False,
        title="Top 15 Products by Basket Share",
    )
    plt.xlabel("Baskets containing product (%)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "retail_top_products.png",
        dpi=200,
    )
    plt.show()

    pd.Series(basket_sizes).value_counts().sort_index().plot(
        kind="bar",
        figsize=(9, 5),
        title="Basket-size Distribution",
    )
    plt.xlabel("Number of products in basket")
    plt.ylabel("Number of baskets")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "retail_basket_size_distribution.png",
        dpi=200,
    )
    plt.show()

    return product_counts


def association_analysis(basket_df, product_counts):
    """Calculate directed pairwise support, confidence and lift."""
    n = len(basket_df)
    min_support_count = int(np.ceil(0.01 * n))

    eligible = product_counts[
        product_counts >= min_support_count
    ].index.tolist()

    assoc = basket_df[eligible]
    rows = []

    for product_a in eligible:
        count_a = int(assoc[product_a].sum())

        for product_b in eligible:
            if product_a == product_b:
                continue

            count_b = int(assoc[product_b].sum())
            joint_count = int(
                (
                    (assoc[product_a] == 1)
                    & (assoc[product_b] == 1)
                ).sum()
            )

            if joint_count >= 75 and count_a and count_b:
                support = joint_count / n
                confidence = joint_count / count_a
                lift = confidence / (count_b / n)

                rows.append((
                    product_a,
                    product_b,
                    joint_count,
                    support,
                    confidence,
                    lift,
                ))

    rules = pd.DataFrame(
        rows,
        columns=[
            "If basket has",
            "Then also",
            "Joint baskets",
            "Support",
            "Confidence",
            "Lift",
        ],
    )

    if not rules.empty:
        rules = rules.sort_values(
            ["Lift", "Confidence"],
            ascending=False,
        )

    rules.to_csv(
        OUTPUT_DIR / "retail_association_rules.csv",
        index=False,
    )

    print("\nTOP 20 ASSOCIATION RULES")
    print(rules.head(20).round(4).to_string(index=False))

    return rules


def clustering_analysis(basket_df):
    """Run PCA and four clustering approaches on a reproducible sample."""
    rng = np.random.RandomState(RANDOM_STATE)
    sample_n = min(2000, len(basket_df))
    indices = rng.choice(
        len(basket_df),
        sample_n,
        replace=False,
    )
    X = basket_df.iloc[indices].to_numpy()

    pca = PCA(random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(X)

    variance = pd.DataFrame({
        "Component": np.arange(
            1, len(pca.explained_variance_ratio_) + 1
        ),
        "Variance explained": pca.explained_variance_ratio_,
        "Cumulative variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })

    variance.to_csv(
        OUTPUT_DIR / "retail_pca_variance.csv",
        index=False,
    )

    print("\nPCA")
    print(
        "First two components variance (%):",
        np.round(
            pca.explained_variance_ratio_[:2] * 100,
            2,
        ),
    )
    print(
        "Cumulative variance first 10 components (%):",
        round(
            pca.explained_variance_ratio_[:10].sum() * 100,
            2,
        ),
    )

    number_to_plot = min(
        10, len(pca.explained_variance_ratio_)
    )
    plt.figure(figsize=(8, 5))
    plt.bar(
        range(1, number_to_plot + 1),
        pca.explained_variance_ratio_[:number_to_plot] * 100,
    )
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained (%)")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "retail_pca_scree.png",
        dpi=200,
    )
    plt.show()

    # K-means validation sweep: k = 2 through 8.
    kmeans_rows = []
    for k in range(2, 9):
        labels = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE,
        ).fit_predict(X)

        kmeans_rows.append((
            k,
            silhouette_score(X, labels),
            davies_bouldin_score(X, labels),
        ))

    kmeans_validation = pd.DataFrame(
        kmeans_rows,
        columns=[
            "k",
            "Silhouette",
            "Davies-Bouldin",
        ],
    )
    kmeans_validation.to_csv(
        OUTPUT_DIR / "retail_kmeans_validation.csv",
        index=False,
    )

    print("\nK-MEANS VALIDATION")
    print(kmeans_validation.round(4).to_string(index=False))

    kmeans_validation.set_index("k").plot(
        marker="o",
        subplots=True,
        figsize=(8, 6),
        title="K-means Validation Scores",
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "retail_kmeans_validation.png",
        dpi=200,
    )
    plt.show()

    cluster_models = {
        "K-means (k=5)": KMeans(
            n_clusters=5,
            n_init=10,
            random_state=RANDOM_STATE,
        ),
        "Hierarchical (complete, k=5)": (
            AgglomerativeClustering(
                n_clusters=5,
                linkage="complete",
            )
        ),
        "Spectral (k=5)": SpectralClustering(
            n_clusters=5,
            affinity="nearest_neighbors",
            n_neighbors=min(15, len(X) - 1),
            random_state=RANDOM_STATE,
        ),
    }

    cluster_rows = []
    for name, model in cluster_models.items():
        labels = model.fit_predict(X)
        cluster_rows.append((
            name,
            len(set(labels)),
            silhouette_score(X, labels),
            davies_bouldin_score(X, labels),
        ))

    # DBSCAN with Jaccard distance on binary basket features.
    distances = pairwise_distances(X, metric="jaccard")
    dbscan_labels = DBSCAN(
        eps=0.5,
        min_samples=5,
        metric="precomputed",
    ).fit_predict(distances)

    non_noise = dbscan_labels != -1
    cluster_count = len(set(dbscan_labels[non_noise]))

    dbscan_silhouette = np.nan
    if non_noise.sum() > 1 and cluster_count > 1:
        dbscan_silhouette = silhouette_score(
            X[non_noise],
            dbscan_labels[non_noise],
        )

    cluster_rows.append((
        "DBSCAN",
        cluster_count,
        dbscan_silhouette,
        np.nan,
    ))

    clustering_metrics = pd.DataFrame(
        cluster_rows,
        columns=[
            "Method",
            "Clusters excluding noise",
            "Silhouette",
            "Davies-Bouldin",
        ],
    )
    clustering_metrics.to_csv(
        OUTPUT_DIR / "retail_clustering_metrics.csv",
        index=False,
    )

    print("\nCLUSTERING COMPARISON")
    print(clustering_metrics.round(4).to_string(index=False))
    print(
        "DBSCAN noise points:",
        int((~non_noise).sum()),
        "out of",
        len(X),
    )

    # Save the PCA projection used for visual inspection.
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_values[:, 0],
        pca_values[:, 1],
        s=12,
        alpha=0.6,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Retail Baskets: PCA Projection")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "retail_pca_projection.png",
        dpi=200,
    )
    plt.show()

    return kmeans_validation, clustering_metrics, variance


def main():
    retail, baskets, basket_df, basket_sizes = load_and_clean_data()

    print("Original dataset shape:", retail.shape)
    print("Number of baskets:", len(baskets))
    print("Distinct products:", basket_df.shape[1])
    print(
        "Basket-product entries:",
        int(basket_df.to_numpy().sum()),
    )

    product_counts = descriptive_analysis(
        basket_df,
        basket_sizes,
    )
    association_analysis(
        basket_df,
        product_counts,
    )
    clustering_analysis(basket_df)

    print(
        "\nRetail analysis complete. CSV tables and plots saved in:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
