import pandas as pd

def load_and_clean_data():
    orders = pd.read_csv("./datasets/olist_orders_dataset.csv")
    items = pd.read_csv("./datasets/olist_order_items_dataset.csv")
    reviews = pd.read_csv("./datasets/olist_order_reviews_dataset.csv")

    orders = orders[orders["order_status"] == "delivered"]

    df_merged = pd.merge(orders, items, on="order_id", how="inner")

    df_final = pd.merge(df_merged, reviews, on="order_id", how="inner")

    util_columns = [
        "customer_id",
        "order_id",
        "order_purchase_timestamp",
        "price",
        "review_score",
        "review_comment_message"
    ]

    df_ml = df_final[util_columns]
    df_ml["order_purchase_timestamp"] = pd.to_datetime(df_ml["order_purchase_timestamp"])

    return df_ml