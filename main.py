from data_prep import load_and_clean_data
from feature_engineering import build_rfm_features

df_base = load_and_clean_data()

df_rfm = build_rfm_features(df_base)

print("RFM Features:")
print(df_rfm)