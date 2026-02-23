import pandas as pd

# Load dataset
df = pd.read_csv("dataset_raw_from_webscraping.csv")
df["user_reviews"] = df["user_reviews"].fillna("")
# Basic inspection (keep but comment noisy parts)
print("Shape:", df.shape)
# print("\nColumns:\n", df.columns.tolist())
# print("\nFirst 5 rows:")
# print(df.head())

# print("\nData types:")
# print(df.dtypes)

# print("\nMissing values:")
# print(df.isnull().sum())

# ----------------------------
# COLUMN TRIAGE
# ----------------------------

columns_to_keep = [
    "title",
    "category",
    "review_count",
    "review_rating",
    "open_hours",
    "status",
    "latitude",
    "longitude",
    "complete_address",
    "user_reviews"
]

df = df[columns_to_keep]

#print("\nAfter column selection:", df.shape)
#print(df.columns)

import re

# ----------------------------
# REVIEW TEXT CLEANING
# ----------------------------

def clean_text(text):
    if pd.isna(text):
        return ""
    
    # convert to lowercase
    text = text.lower()
    
    # remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# apply cleaning
df["clean_reviews"] = df["user_reviews"].apply(clean_text)

#print("\nSample cleaned reviews:")
#print(df["clean_reviews"].head(5))

#print(df["user_reviews"].iloc[0])

import json

# ----------------------------
# EXTRACT REAL REVIEW TEXT
# ----------------------------

def extract_descriptions(review_blob):
    if pd.isna(review_blob):
        return ""
    
    try:
        reviews = json.loads(review_blob)
        descriptions = [r.get("Description", "") for r in reviews]
        return " ".join(descriptions)
    except:
        return ""

df["review_text"] = df["user_reviews"].apply(extract_descriptions)

#print("\nSample extracted review text:")
#print(df["review_text"].head(3))

# ----------------------------
# FINAL TEXT NORMALIZATION
# ----------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["review_clean"] = df["review_text"].apply(normalize_text)

#print("\nNormalized review sample:")
#print(df["review_clean"].head(3))

from collections import Counter

# ----------------------------
# WORD FREQUENCY (FILTERED)
# ----------------------------

all_words = " ".join(df["review_clean"]).split()

# Indonesian stopwords (basic set)
indo_stopwords = [
    "dan","di","yang","yg","nya","saya","ada","ga","tidak",
    "juga","itu","ini","untuk","dengan","ke","dari","karena",
    "the","it","my","is","in","to","of"
]

filtered_words = [
    w for w in all_words
    if w not in indo_stopwords and len(w) > 2
]

word_freq = Counter(filtered_words)

#print("\nTop 20 most common words (filtered):")
#print(word_freq.most_common(20))
# ----------------------------
# TOPIC MODELING (THEMES)
# ----------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# remove empty reviews
texts = df["review_clean"].dropna()
texts = texts[texts.str.strip() != ""]

# convert text to document-term matrix
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    stop_words=indo_stopwords
)

dtm = vectorizer.fit_transform(texts)

# build LDA model
lda = LatentDirichletAllocation(
    n_components=4,
    random_state=42
)

lda.fit(dtm)

# show topics
words = vectorizer.get_feature_names_out()

print("\n=== Discovered Review Themes ===")

for idx, topic in enumerate(lda.components_):
    print(f"\nTheme {idx+1}:")
    top_words = [words[i] for i in topic.argsort()[-10:]]
    print(", ".join(top_words))

# ----------------------------
# BUSINESS PERFORMANCE SCORE
# ----------------------------

# avoid division issues
df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce").fillna(0)

# simple weighted score
df["performance_score"] = df["review_rating"] * (df["review_count"] ** 0.5) 

# sort best businesses
top_businesses = df.sort_values(
    by="performance_score",
    ascending=False
)[["title", "review_rating", "review_count", "performance_score"]]

print("\n=== Top Performing Laundry Businesses ===")
print(top_businesses.head(10))

# ----------------------------
# GEOGRAPHIC VISUALIZATION (TOP SHOPS ONLY)
# ----------------------------
import matplotlib.pyplot as plt

# Get top shops (example: by number of reviews)
top_shops = df.sort_values(by="performance_score", ascending=False).head(20)

plt.figure(figsize=(8,6))

plt.scatter(
    top_shops["longitude"],
    top_shops["latitude"],
    s=top_shops["performance_score"] * 0.5,
    alpha=0.7
)

plt.title("Top Laundry Business Locations (Bandung)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()