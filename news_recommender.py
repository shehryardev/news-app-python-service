import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import os

# ---------------------------
# MongoDB Config & Connection
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "news_db_dev"
NEWS_COLLECTION = "news"
LIKES_COLLECTION = "likes"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
news_col = db[NEWS_COLLECTION]
likes_col = db[LIKES_COLLECTION]

# Fetch all news documents (include _id!)
cursor = news_col.find({}, {
    "_id": 1,
    "title": 1,
    "tags": 1
})

# Convert to DataFrame
news_data = pd.DataFrame(list(cursor))
news_data["_id"] = news_data["_id"].astype(str)  # Convert ObjectId to string

# Preview
print(news_data.head())

# ---------------------------
# SentenceTransformer Embedding
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Clean tag text
news_data["tags_text"] = news_data["tags"].apply(
    lambda tags: " ".join(tags) if isinstance(tags, list) else str(tags)
)

# Encode tags into embeddings
news_data["embedding"] = model.encode(
    news_data["tags_text"].tolist(),
    convert_to_tensor=False
).tolist()

# ---------------------------
# Recommendation Function
# ---------------------------
def get_user_likes(user_id):
    """
    Fetch liked news article URLs for a user from MongoDB likes collection.
    """
    likes = likes_col.find({"user_id": user_id})
    return [like["news_url"] for like in likes]

def recommend_for_user(user_id, top_n=10):
    liked_urls = get_user_likes(user_id)
    if not liked_urls:
        print("No liked articles found for this user.")
        return pd.DataFrame()

    liked_embeddings = news_data[news_data["_id"].isin(liked_urls)]["embedding"].tolist()
    if not liked_embeddings:
        print("No matching embeddings found for user's liked articles.")
        return pd.DataFrame()

    user_profile = np.mean(liked_embeddings, axis=0).reshape(1, -1)
    all_embeddings = np.stack(news_data["embedding"].to_numpy())
    similarities = cosine_similarity(user_profile, all_embeddings).flatten()

    news_data["similarity"] = similarities
    recommendations = news_data[~news_data["_id"].isin(liked_urls)] \
        .sort_values(by="similarity", ascending=False) \
        .head(top_n)

    return recommendations[["_id", "title", "tags", "similarity"]]

# ---------------------------
# CLI Tester
# ---------------------------
if __name__ == "__main__":
    print("Enter user_id (MongoDB _id as string): ")
    user_id = input().strip()
    try:
        recs = recommend_for_user(user_id)
        if not recs.empty:
            print("\nRecommended articles:")
            print(recs.to_string(index=False))
        else:
            print("No recommendations found.")
    except Exception as e:
        print("Error:", str(e))
