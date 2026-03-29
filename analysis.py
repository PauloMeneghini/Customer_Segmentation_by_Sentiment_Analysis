def analyze_clusters(df_rfm):
    profile = df_rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    print("\nCluster Profiles:")
    print(profile)
    
    