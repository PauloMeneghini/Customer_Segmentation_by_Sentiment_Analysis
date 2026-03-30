def analyze_clusters(df_rfm):
    profile = df_rfm.groupby("cluster")[["recency", "frequency", "monetary", "avg_sentiment"]].mean()
    
    names = {0: "Regular", 1: "At Risk", 2: "VIP"}
    df_rfm["segment"] = df_rfm["cluster"].map(names)

    print("\nCustomer Segments:")
    print(df_rfm["segment"].value_counts())

    print("\nAverages cluster")
    # Formatando para ficar fácil de ler no terminal
    print(profile.round(2))
    
    print("\nCluster size:")
    print(df_rfm["cluster"].value_counts())
    
    