from data_prep import load_and_clean_data
from feature_engineering import build_rfm_features
from train_model import scale_features, find_best_k, train_kmeans
from analysis import analyze_clusters
from nlp_processing import extract_sentiment_features

df_base = load_and_clean_data()

df_nlp = extract_sentiment_features(df_base)

df_rfm = build_rfm_features(df_base, df_nlp)

print("\nRFM Features:")
print(df_rfm)

df_scaled = scale_features(df_rfm)

find_best_k(df_scaled)

df_rfm = train_kmeans(df_rfm, df_scaled, k=3)

print("\nRFM Features with Clusters:")
print(df_rfm)

analyze_clusters(df_rfm)