from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def scale_features(df_rfm):
    columns_to_scale = ["recency", "frequency", "monetary"]

    df_scaled = StandardScaler().fit_transform(df_rfm[columns_to_scale])

    df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)

    print(df_scaled.head())
    return df_scaled

def find_best_k(df_scaled):
    inertias = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.savefig("elbow_method.png")
    print("Elbow method saved to elbow_method.png")
    plt.show()

