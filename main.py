from data_prep import load_and_clean_data
from feature_engineering import build_rfm_features
from train_model import scale_features, find_best_k

df_base = load_and_clean_data()

df_rfm = build_rfm_features(df_base)

print("RFM Features:")
print(df_rfm)

df_scaled = scale_features(df_rfm)

find_best_k(df_scaled)