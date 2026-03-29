def analyze_clusters(df_rfm):
    profile = df_rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    names = {0: "Regular", 1: "At Risk", 2: "VIP"}
    df_rfm["segment"] = df_rfm["cluster"].map(names)
    print("\nCustomer Segments:")
    print(df_rfm["segment"].value_counts())
    
    