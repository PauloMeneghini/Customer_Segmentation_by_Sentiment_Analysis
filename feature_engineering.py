import pandas as pd

def build_rfm_features(df_ml):
    last_date = df_ml["order_purchase_timestamp"].max()
    recent_date = last_date + pd.Timedelta(days=1)

    recency_df = df_ml.groupby("customer_id")["order_purchase_timestamp"].max().reset_index()

    recency_df.columns = ["customer_id", "last_purchase_date"]

    recency_df["recency"] = (recent_date - recency_df["last_purchase_date"]).dt.days

    frequency_df = df_ml.groupby("customer_id")["order_id"].nunique().reset_index()

    frequency_df.columns = ["customer_id", "frequency"]

    monetary_df = df_ml.groupby("customer_id")["price"].sum().reset_index()

    monetary_df.columns = ["customer_id", "monetary"]

    df_rfm = recency_df.merge(frequency_df, on="customer_id", how="inner").merge(monetary_df, on="customer_id", how="inner")

    return df_rfm

    