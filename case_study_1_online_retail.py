"""Case Study 1: Online Retail basket analysis. Run from the repository folder."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
path = DATA / "Online_Retail.csv"
if not path.exists():
    raise FileNotFoundError(f"Place the retail CSV at {path}. See README for source details.")
retail = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
baskets = [sorted(set(str(v).strip().lower() for v in row if str(v).strip())) for row in retail.to_numpy()]
sizes = np.array([len(b) for b in baskets])
products = sorted({p for b in baskets for p in b})
product_to_col = {p:i for i,p in enumerate(products)}
mat = np.zeros((len(baskets), len(products)), dtype=np.uint8)
for i,b in enumerate(baskets):
    for p in b: mat[i, product_to_col[p]] = 1
basket_df = pd.DataFrame(mat, columns=products)
summary = pd.DataFrame([{"Baskets":len(baskets),"Distinct products":len(products),"Basket-product entries":int(mat.sum()),"Mean basket size":round(float(sizes.mean()),3),"Median basket size":float(np.median(sizes)),"Single-item basket share (%)":round(float((sizes==1).mean()*100),2)}])
summary.to_csv(OUT/"retail_summary.csv",index=False)
counts=basket_df.sum().sort_values(ascending=False)
counts.rename("Basket count").to_frame().assign(**{"Basket share (%)":(counts/len(basket_df)*100).round(2)}).to_csv(OUT/"retail_top_products.csv")
n=len(basket_df); eligible=counts[counts>=int(np.ceil(.01*n))].index.tolist(); assoc=basket_df[eligible]; rows=[]
for a in eligible:
    ac=int(assoc[a].sum())
    for b in eligible:
        if a==b: continue
        bc=int(assoc[b].sum()); joint=int(((assoc[a]==1)&(assoc[b]==1)).sum())
        if joint>=75 and ac and bc:
            conf=joint/ac; rows.append((a,b,joint,joint/n,conf,conf/(bc/n)))
pd.DataFrame(rows,columns=["If basket has","Then also","Joint baskets","Support","Confidence","Lift"]).sort_values(["Lift","Confidence"],ascending=False).to_csv(OUT/"retail_association_rules.csv",index=False)
# Reproducible clustering comparison on a sample
rng=np.random.RandomState(42); ix=rng.choice(len(basket_df),min(2000,len(basket_df)),replace=False); X=basket_df.iloc[ix].to_numpy()
pca=PCA(random_state=42).fit(X)
cluster_rows=[]
for k in range(2,9):
    labels=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(X)
    cluster_rows.append((k,silhouette_score(X,labels),davies_bouldin_score(X,labels)))
pd.DataFrame(cluster_rows,columns=["k","Silhouette","Davies-Bouldin"]).to_csv(OUT/"retail_clustering_metrics.csv",index=False)
pd.DataFrame({"component":np.arange(1,len(pca.explained_variance_ratio_)+1),"variance_explained":pca.explained_variance_ratio_}).to_csv(OUT/"retail_pca_variance.csv",index=False)
print("Retail analysis complete. Tables saved in outputs/.")
