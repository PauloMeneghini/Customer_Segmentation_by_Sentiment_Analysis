import pandas as pd
import re  
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r'[^a-záéíóúâêîôûãõç\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_sentiment_features(df_ml):
    print("=" * 50)
    print("\nExtracting sentiment features...")

    df_nlp = df_ml.copy()

    df_nlp["review_comment_message"] = df_nlp["review_comment_message"].fillna("")

    print("=" * 50)
    print("Cleaning characters and scores...")
    df_nlp["clean_review"] = df_nlp["review_comment_message"].apply(clean_text)

    print("=" * 50)
    print("Extracting sentiment scores...")
    sentiment_map = {
        1: -1.0,  # Muito Negativo
        2: -0.5,  # Negativo
        3: 0.0,   # Neutro
        4: 0.5,   # Positivo
        5: 1.0    # Muito Positivo
    }
    # sentiment_map = {
    #     1: "negative", 
    #     2: "negative", 
    #     3: "neutral", 
    #     4: "positive", 
    #     5: "positive"
    # }

    df_nlp["sentiment_score"] = df_nlp["review_score"].map(sentiment_map)

    return df_nlp[["customer_id", "sentiment_score", "clean_review"]]
    

    

    
