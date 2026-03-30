import pandas as pd

def build_rfm_features(df_ml):
    last_date = df_ml["order_purchase_timestamp"].max()
    recent_date = last_date + pd.Timedelta(days=1)

    df_rfm = df_ml.groupby("customer_id").agg(
        last_purchase_date = ("order_purchase_timestamp", "max"),
        frequency = ("order_id", "nunique"),
        monetary = ("price", "sum")
    ).reset_index()

    df_rfm["recency"] = (recent_date - df_rfm["last_purchase_date"]).dt.days

    df_rfm = df_rfm[["customer_id", "recency", "frequency", "monetary"]]

    return df_rfm

    