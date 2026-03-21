import pandas as pd
import re
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_descriptions(review_blob):
    if pd.isna(review_blob) or review_blob == "":
        return ""
    try:
        # Check if it's already a list/dict or a stringified JSON
        if isinstance(review_blob, str):
            reviews = json.loads(review_blob)
        else:
            reviews = review_blob
            
        descriptions = [r.get("Description", "") for r in reviews if isinstance(r, dict)]
        return " ".join(descriptions)
    except:
        return ""

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_data(file_source):
    """
    Main engine function that takes a file path or file-like object 
    and returns the processed DataFrame and discovered topics.
    """
    # 1. Load Data
    df = pd.read_csv(file_source)
    
    # 2. Basic Cleaning & Column Selection
    columns_to_keep = [
        "title", "category", "review_count", "review_rating",
        "latitude", "longitude", "complete_address", "user_reviews"
    ]
    # Ensure columns exist before filtering
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[existing_cols].copy()
    
    # 3. Text Processing Pipeline
    df["user_reviews"] = df["user_reviews"].fillna("")
    df["review_text"] = df["user_reviews"].apply(extract_descriptions)
    df["review_clean"] = df["review_text"].apply(normalize_text)
    
    # 4. Scoring Logic
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce").fillna(0)
    # Ensure coordinates are numeric for the map
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    # Weighted Score: Rating * Square Root of Volume
    df["performance_score"] = df["review_rating"] * (df["review_count"] ** 0.5)
    
    # 5. Topic Modeling (LDA)
    indo_stopwords = [
        "dan","di","yang","yg","nya","saya","ada","ga","tidak",
        "juga","itu","ini","untuk","dengan","ke","dari","karena",
        "the","it","my","is","in","to","of", "laundry", "cuci", "bersih"
    ]
    
    # Filter out empty strings for LDA
    texts = df["review_clean"].dropna()
    texts = texts[texts.str.strip() != ""]
    
    topics_list = []
    
    if not texts.empty:
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words=indo_stopwords
        )
        
        try:
            dtm = vectorizer.fit_transform(texts)
            lda = LatentDirichletAllocation(n_components=4, random_state=42)
            lda.fit(dtm)
            
            words = vectorizer.get_feature_names_out()
            for topic in lda.components_:
                top_words = [words[i] for i in topic.argsort()[-8:]] # Top 8 words
                topics_list.append(", ".join(top_words))
        except ValueError:
            # Handle cases where vocabulary is too small
            topics_list = ["Insufficient text data for topic modeling"] * 4
    else:
        topics_list = ["No reviews found to analyze"] * 4

    # Sort by performance before returning
    df = df.sort_values(by="performance_score", ascending=False)
    
    return df, topics_list