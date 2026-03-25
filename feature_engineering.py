import pandas as pd

def build_rfm_features(df_ml):
    last_date = df_ml["order_purchase_timestamp"].max()
    recent_date = last_date + pd.Timedelta(days=1)

    client_grouped = df_ml.groupby("customer_id")["order_purchase_timestamp"].max().reset_index()

    client_grouped.columns = ["customer_id", "last_purchase_date"]

    client_grouped["recency"] = (recent_date - client_grouped["last_purchase_date"]).dt.days

    print(client_grouped["recency"])

    return client_grouped[["customer_id", "recency"]]