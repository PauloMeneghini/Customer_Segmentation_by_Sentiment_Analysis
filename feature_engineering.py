import pandas as pd

def build_rfm_features(df_ml, df_nlp):
    last_date = df_ml["order_purchase_timestamp"].max()
    recent_date = last_date + pd.Timedelta(days=1)

    df_completo = pd.merge(
        df_ml, 
        df_nlp[['customer_id', 'sentiment_score']], 
        on='customer_id', 
        how='left'
    )

    df_rfm = df_completo.groupby("customer_id").agg(
        last_purchase_date=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum"),
        avg_sentiment=("sentiment_score", "mean")
    ).reset_index()

    df_rfm["recency"] = (recent_date - df_rfm["last_purchase_date"]).dt.days

    limite = df_rfm["monetary"].quantile(0.99)
    df_rfm = df_rfm[df_rfm["monetary"] < limite]

    return df_rfm[["customer_id", "recency", "frequency", "monetary", "avg_sentiment"]]