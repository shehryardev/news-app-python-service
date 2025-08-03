from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
import random
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import json

# ========== CONFIG ==========
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "news_db_dev"
USERS_COLLECTION = "users"
LIKES_COLLECTION = "likes"
SEEN_COLLECTION = "seen_articles"

# Google OAuth Config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id.googleusercontent.com")

# ========== DB SETUP ==========
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db[USERS_COLLECTION]
likes_col = db[LIKES_COLLECTION]
seen_col = db[SEEN_COLLECTION]

# users_col.create_index("email", unique=True)
# likes_col.create_index([("user_id", 1), ("news_url", 1)], unique=True)
# seen_col.create_index([("user_id", 1), ("news_id", 1)], unique=True)

# ========== SECURITY ==========
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# ========== SCHEMAS ==========
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class GoogleAuthRequest(BaseModel):
    token: str

class UserOut(BaseModel):
    id: str = Field(..., alias="_id")
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
    auth_provider: Optional[str] = "email"

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

class LikeIn(BaseModel):
    news_id: str

class LikeOut(BaseModel):
    news_id: str
    liked_at: datetime

class SeenIn(BaseModel):
    news_ids: List[str]

class SeenOut(BaseModel):
    news_id: str
    seen_at: datetime

# ========== UTILS ==========
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_email(email: str):
    return users_col.find_one({"email": email})

def get_user(user_id: str):
    return users_col.find_one({"_id": ObjectId(user_id)})

def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password"]):
        return None
    return user

def verify_google_token(token: str):
    """Verify Google OAuth token and return user info"""
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        # Verify the issuer
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        return {
            'email': idinfo['email'],
            'name': idinfo.get('name'),
            'picture': idinfo.get('picture'),
            'google_id': idinfo['sub']
        }
    except ValueError as e:
        print(f"Token verification failed: {e}")
        return None

def create_or_get_google_user(google_user_info):
    """Create a new user or get existing user for Google OAuth"""
    email = google_user_info['email']
    existing_user = get_user_by_email(email)
    
    if existing_user:
        # Update user info if it's a Google user or convert to Google user
        update_data = {
            "name": google_user_info.get('name'),
            "picture": google_user_info.get('picture'),
            "auth_provider": "google",
            "google_id": google_user_info['google_id']
        }
        users_col.update_one({"_id": existing_user["_id"]}, {"$set": update_data})
        existing_user.update(update_data)
        return existing_user
    else:
        # Create new Google user
        user_doc = {
            "email": email,
            "name": google_user_info.get('name'),
            "picture": google_user_info.get('picture'),
            "auth_provider": "google",
            "google_id": google_user_info['google_id'],
            "password": None  # No password for Google users
        }
        result = users_col.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.user_id)
    if user is None:
        raise credentials_exception
    return user

# ========== RECOMMENDATION UTILS ==========
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_user_likes_ids(user_id):
    likes = likes_col.find({"user_id": user_id})
    return [like["news_id"] for like in likes]

def get_user_seen_ids(user_id):
    seen = seen_col.find({"user_id": user_id})
    return [seen_item["news_id"] for seen_item in seen]

def mark_articles_as_seen(user_id, news_ids):
    """Mark multiple articles as seen by the user"""
    seen_docs = []
    current_time = datetime.utcnow()
    for news_id in news_ids:
        seen_docs.append({
            "user_id": user_id,
            "news_id": news_id,
            "seen_at": current_time
        })
    if seen_docs:
        try:
            seen_col.insert_many(seen_docs, ordered=False)  # ordered=False to continue on duplicates
        except Exception as e:
            # Ignore duplicate key errors
            pass

def get_news_df():
    cursor = db["news"].find({}, {"_id": 1, "title": 1, "tags": 1, "created_at": 1, "published_at": 1})
    news_data = pd.DataFrame(list(cursor))
    if news_data.empty:
        return news_data
    news_data["_id"] = news_data["_id"].astype(str)
    news_data["tags_text"] = news_data["tags"].apply(
        lambda tags: " ".join(tags) if isinstance(tags, list) else str(tags)
    )
    news_data["embedding"] = model.encode(
        news_data["tags_text"].tolist(),
        convert_to_tensor=False
    ).tolist()
    
    # Handle date fields - use published_at if available, otherwise created_at
    news_data["date"] = news_data.apply(
        lambda row: row.get("published_at") or row.get("created_at") or datetime.utcnow(),
        axis=1
    )
    return news_data

def recommend_for_user(user_id, top_n=10, random_ratio=0.3, sort_by_date=True, min_similar_articles=2):
    try:
        news_data = get_news_df()
        if news_data.empty:
            return []
        liked_ids = get_user_likes_ids(user_id)
        seen_ids = get_user_seen_ids(user_id)
        
        # Combine liked and seen articles to exclude
        excluded_ids = list(set(liked_ids + seen_ids))
        
        # If user has no interaction history, return diverse random articles
        if not liked_ids:
            available_articles = news_data[~news_data["_id"].isin(excluded_ids)].copy()
            
            if available_articles.empty:
                return []
            
            # For new users, provide truly random diverse recommendations
            # Try to get articles from different categories/tags for variety
            result = []
            
            # If we have enough articles, try to diversify by tags
            if len(available_articles) >= top_n * 2:
                # Get unique tags from all available articles
                all_tags = set()
                for tags in available_articles["tags"]:
                    if isinstance(tags, list):
                        all_tags.update(tags)
                    elif isinstance(tags, str):
                        all_tags.add(tags)
                
                # Try to pick articles from different tag categories
                selected_articles = []
                used_tags = set()
                
                # First pass: try to get articles with diverse tags
                for _, row in available_articles.sample(n=len(available_articles)).iterrows():
                    if len(selected_articles) >= top_n:
                        break
                    
                    article_tags = row["tags"] if isinstance(row["tags"], list) else [row["tags"]]
                    # Check if this article introduces new tags
                    new_tags = set(article_tags) - used_tags
                    
                    if new_tags or len(selected_articles) < top_n // 2:
                        selected_articles.append(row)
                        used_tags.update(article_tags)
                
                # If we still need more articles, fill randomly
                if len(selected_articles) < top_n:
                    remaining_articles = available_articles[~available_articles["_id"].isin([a["_id"] for a in selected_articles])]
                    additional_count = top_n - len(selected_articles)
                    if not remaining_articles.empty:
                        additional_articles = remaining_articles.sample(n=min(additional_count, len(remaining_articles)))
                        selected_articles.extend(additional_articles.to_dict('records'))
                
                # Convert to the expected format
                for row in selected_articles[:top_n]:
                    result.append({
                        "_id": row["_id"],
                        "title": row["title"],
                        "tags": row["tags"],
                        "similarity": 0.0  # Mark as random
                    })
            else:
                # If not enough articles for diversity, just pick randomly
                random_articles = available_articles.sample(n=min(top_n, len(available_articles)))
                for _, row in random_articles.iterrows():
                    result.append({
                        "_id": row["_id"],
                        "title": row["title"],
                        "tags": row["tags"],
                        "similarity": 0.0  # Mark as random
                    })
            
            # Final shuffle for good measure
            random.shuffle(result)
            return result[:top_n]
        
        # Filter out already liked and seen articles
        available_news = news_data[~news_data["_id"].isin(excluded_ids)].copy()
        if available_news.empty:
            return []
        
        # Calculate similarities first
        liked_embeddings = news_data[news_data["_id"].isin(liked_ids)]["embedding"].tolist()
        if not liked_embeddings:
            return []
            
        user_profile = np.mean(liked_embeddings, axis=0).reshape(1, -1)
        all_embeddings = np.stack(available_news["embedding"].to_numpy())
        similarities = cosine_similarity(user_profile, all_embeddings).flatten()
        available_news["similarity"] = similarities
        
        # Ensure we get at least min_similar_articles with high similarity
        highly_similar = []
        similarity_thresholds = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        for threshold in similarity_thresholds:
            highly_similar_candidates = available_news[available_news["similarity"] >= threshold]
            if len(highly_similar_candidates) >= min_similar_articles:
                # Get the required number of highly similar articles
                if sort_by_date and len(highly_similar_candidates) > min_similar_articles:
                    # From highly similar, pick the most recent ones
                    highly_similar = highly_similar_candidates.nlargest(min_similar_articles * 2, "date").nlargest(min_similar_articles, "similarity")
                else:
                    highly_similar = highly_similar_candidates.nlargest(min_similar_articles, "similarity")
                break
        
        # If we still don't have enough, take what we can get
        if len(highly_similar) < min_similar_articles:
            highly_similar = available_news.nlargest(min(min_similar_articles, len(available_news)), "similarity")
        
        recommendations = []
        used_ids = set()
        
        # Add highly similar articles
        for _, row in highly_similar.iterrows():
            recommendations.append({
                "_id": row["_id"],
                "title": row["title"],
                "tags": row["tags"],
                "similarity": row["similarity"]
            })
            used_ids.add(row["_id"])
        
        # Calculate remaining slots
        remaining_slots = top_n - len(recommendations)
        if remaining_slots <= 0:
            # Shuffle and return
            random.shuffle(recommendations)
            return recommendations[:top_n]
        
        # Filter out already used articles
        remaining_news = available_news[~available_news["_id"].isin(used_ids)].copy()
        
        if remaining_news.empty:
            # Shuffle and return what we have
            random.shuffle(recommendations)
            return recommendations
        
        # Calculate personalized vs random for remaining slots
        personalized_count = int(remaining_slots * (1 - random_ratio))
        random_count = remaining_slots - personalized_count
        
        # Get additional personalized recommendations
        if personalized_count > 0:
            if sort_by_date and len(remaining_news) > personalized_count * 2:
                # Get more candidates than needed, sort by date, then take top by similarity
                candidate_count = min(personalized_count * 3, len(remaining_news))
                recent_candidates = remaining_news.nlargest(candidate_count, "date")
                personalized = recent_candidates.nlargest(personalized_count, "similarity")
            else:
                # Traditional similarity-only approach
                personalized = remaining_news.nlargest(personalized_count, "similarity")
            
            for _, row in personalized.iterrows():
                recommendations.append({
                    "_id": row["_id"],
                    "title": row["title"],
                    "tags": row["tags"],
                    "similarity": row["similarity"]
                })
                used_ids.add(row["_id"])
            
            # Remove personalized articles from remaining pool
            remaining_news = remaining_news[~remaining_news["_id"].isin(used_ids)]
        
        # Get random recommendations from remaining articles
        if random_count > 0 and not remaining_news.empty:
            if sort_by_date and len(remaining_news) > random_count * 2:
                # When sorting by date, pick random articles from the more recent half
                recent_half = remaining_news.nlargest(len(remaining_news) // 2, "date")
                random_sample = recent_half.sample(n=min(random_count, len(recent_half)))
            else:
                # Standard random selection
                random_sample = remaining_news.sample(n=min(random_count, len(remaining_news)))
            
            for _, row in random_sample.iterrows():
                recommendations.append({
                    "_id": row["_id"],
                    "title": row["title"],
                    "tags": row["tags"],
                    "similarity": 0.0  # Mark as random
                })
        
        # IMPORTANT: Shuffle the final recommendations for variety
        # This ensures highly similar articles aren't clustered together
        random.shuffle(recommendations)
        
        return recommendations[:top_n]
    except Exception as e:
        print(f"Error in recommend_for_user: {e}")
        return []

# ========== FASTAPI APP ==========
app = FastAPI()

# Allow all origins in CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register", response_model=UserOut)
def register(user: UserCreate):
    if get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(user.password)
    user_doc = {
        "email": user.email, 
        "password": hashed_pw, 
        "auth_provider": "email",
        "name": None,
        "picture": None
    }
    result = users_col.insert_one(user_doc)
    user_doc["_id"] = str(result.inserted_id)
    return UserOut(**user_doc)

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": str(user["_id"])})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/google", response_model=Token)
def google_auth(auth_request: GoogleAuthRequest):
    """Authenticate user with Google OAuth token"""
    google_user_info = verify_google_token(auth_request.token)
    
    if not google_user_info:
        raise HTTPException(
            status_code=400, 
            detail="Invalid Google token"
        )
    
    user = create_or_get_google_user(google_user_info)
    access_token = create_access_token(data={"sub": str(user["_id"])})
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserOut)
def get_current_user_profile(current_user=Depends(get_current_user)):
    """Get current user profile"""
    user_dict = dict(current_user)
    user_dict["_id"] = str(user_dict["_id"])
    return UserOut(**user_dict)

@app.post("/like", response_model=LikeOut)
def like_article(like: LikeIn, current_user=Depends(get_current_user)):
    like_doc = {
        "user_id": str(current_user["_id"]),
        "news_id": like.news_id,
        "liked_at": datetime.utcnow()
    }
    try:
        likes_col.insert_one(like_doc)
    except Exception:
        pass
    return LikeOut(news_id=like.news_id, liked_at=like_doc["liked_at"])

@app.delete("/like", response_model=dict)
def unlike_article(like: LikeIn, current_user=Depends(get_current_user)):
    result = likes_col.delete_one({
        "user_id": str(current_user["_id"]),
        "news_id": like.news_id
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Like not found")
    return {"detail": "Unliked"}

@app.get("/likes", response_model=List[LikeOut])
def get_likes(current_user=Depends(get_current_user)):
    likes = likes_col.find({"user_id": str(current_user["_id"])})
    return [LikeOut(news_id=like["news_id"], liked_at=like["liked_at"]) for like in likes]

@app.get("/recommendations")
def get_recommendations(
    current_user=Depends(get_current_user),
    top_n: int = 10,
    trending: bool = Query(False, description="If true, return trending articles instead of personalized recommendations"),
    days: int = Query(7, description="Number of days to consider for trending articles"),
    random_ratio: float = Query(0.3, ge=0.0, le=1.0, description="Ratio of random articles to include (0.0 = all personalized, 1.0 = all random)"),
    sort_by_date: bool = Query(True, description="If true, prioritize recent articles in personalized recommendations"),
    min_similar_articles: int = Query(2, ge=0, le=10, description="Minimum number of highly similar articles to include (tries thresholds 0.7, 0.6, 0.5, etc.)")
):
    try:
        if trending:
            since = datetime.utcnow() - timedelta(days=days)
            pipeline = [
                {"$match": {"liked_at": {"$gte": since}}},
                {"$group": {"_id": "$news_id", "like_count": {"$sum": 1}}},
                {"$sort": {"like_count": -1}},
                {"$limit": top_n}
            ]
            trending_likes = list(likes_col.aggregate(pipeline))
            news_ids = [item["_id"] for item in trending_likes]
            news_docs = list(db["news"].find({"_id": {"$in": [ObjectId(i) for i in news_ids]}}))
            like_count_map = {item["_id"]: item["like_count"] for item in trending_likes}
            for doc in news_docs:
                doc["_id"] = str(doc["_id"])
                doc["like_count"] = like_count_map.get(str(doc["_id"]), 0)
            return news_docs
        else:
            user_id = str(current_user["_id"])
            recs = recommend_for_user(user_id, top_n=top_n, random_ratio=random_ratio, sort_by_date=sort_by_date, min_similar_articles=min_similar_articles)
            rec_ids = [r["_id"] for r in recs]
            
            # Automatically mark these articles as seen
            if rec_ids:
                mark_articles_as_seen(user_id, rec_ids)
            
            news_docs = list(db["news"].find({"_id": {"$in": [ObjectId(i) for i in rec_ids]}}))
            sim_map = {r["_id"]: r["similarity"] for r in recs}
            # Get like counts for these articles
            like_counts = list(likes_col.aggregate([
                {"$match": {"news_id": {"$in": rec_ids}}},
                {"$group": {"_id": "$news_id", "like_count": {"$sum": 1}}}
            ]))
            like_count_map = {item["_id"]: item["like_count"] for item in like_counts}
            for doc in news_docs:
                doc["_id"] = str(doc["_id"])
                doc["similarity"] = sim_map.get(doc["_id"], None)
                doc["like_count"] = like_count_map.get(doc["_id"], 0)
                doc["is_random"] = sim_map.get(doc["_id"], 0) == 0.0  # Mark random articles
                doc["is_highly_similar"] = sim_map.get(doc["_id"], 0) >= 0.4  # Mark highly similar articles
            return news_docs
    except Exception as e:
        print(f"Error in recommend_for_user: {e}")
        return []

@app.post("/mark-seen", response_model=dict)
def mark_articles_seen(seen: SeenIn, current_user=Depends(get_current_user)):
    """Manually mark articles as seen by the user"""
    user_id = str(current_user["_id"])
    mark_articles_as_seen(user_id, seen.news_ids)
    return {"detail": f"Marked {len(seen.news_ids)} articles as seen"}

@app.get("/seen", response_model=List[SeenOut])
def get_seen_articles(current_user=Depends(get_current_user), limit: int = Query(100, ge=1, le=1000)):
    """Get list of articles seen by the user"""
    user_id = str(current_user["_id"])
    seen_articles = seen_col.find({"user_id": user_id}).sort("seen_at", -1).limit(limit)
    return [SeenOut(news_id=seen["news_id"], seen_at=seen["seen_at"]) for seen in seen_articles]

@app.delete("/seen", response_model=dict)
def clear_seen_articles(current_user=Depends(get_current_user)):
    """Clear all seen articles for the user (useful for testing or reset)"""
    user_id = str(current_user["_id"])
    result = seen_col.delete_many({"user_id": user_id})
    return {"detail": f"Cleared {result.deleted_count} seen articles"}

@app.get("/user-stats")
def get_user_stats(current_user=Depends(get_current_user)):
    """Get user interaction statistics"""
    user_id = str(current_user["_id"])
    liked_count = likes_col.count_documents({"user_id": user_id})
    seen_count = seen_col.count_documents({"user_id": user_id})
    total_news = db["news"].count_documents({})
    
    return {
        "liked_articles": liked_count,
        "seen_articles": seen_count,
        "total_articles": total_news,
        "unseen_articles": max(0, total_news - seen_count),
        "coverage_percentage": round((seen_count / total_news * 100), 2) if total_news > 0 else 0
    }

@app.get("/user-tags")
def get_user_tags(current_user=Depends(get_current_user), top_n: int = 10):
    user_id = str(current_user["_id"])
    liked_ids = get_user_likes_ids(user_id)
    if not liked_ids:
        return {"tags": []}
    # Get tags from liked articles
    cursor = db["news"].find({"_id": {"$in": [ObjectId(i) for i in liked_ids]}}, {"tags": 1})
    tag_counter = Counter()
    for doc in cursor:
        tags = doc.get("tags", [])
        if isinstance(tags, list):
            tag_counter.update(tags)
        elif isinstance(tags, str):
            tag_counter.update([tags])
    # Return top N tags
    most_common = tag_counter.most_common(top_n)
    return {"tags": [{"tag": tag, "count": count} for tag, count in most_common]}

@app.get("/articles")
def get_articles(
    search: str = Query(None, description="Search term for title or tags"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
    exclude_seen: bool = Query(True, description="Exclude articles already seen by the user"),
    current_user=Depends(get_current_user)
):
    query = {}
    if search:
        query = {
            "$or": [
                {"title": {"$regex": search, "$options": "i"}},
                {"tags": {"$elemMatch": {"$regex": search, "$options": "i"}}}
            ]
        }
    
    # Exclude seen articles if requested
    if exclude_seen:
        user_id = str(current_user["_id"])
        seen_ids = get_user_seen_ids(user_id)
        if seen_ids:
            query["_id"] = {"$nin": [ObjectId(id) for id in seen_ids]}
    
    cursor = db["news"].find(query).sort("created_at", -1).skip(skip).limit(limit)
    articles = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        articles.append(doc)
    
    # Optionally mark browsed articles as seen
    if articles and exclude_seen:
        article_ids = [article["_id"] for article in articles]
        mark_articles_as_seen(str(current_user["_id"]), article_ids)
    
    return {"results": articles, "skip": skip, "limit": limit, "count": len(articles)}

# Export the app for Vercel
# The FastAPI app is already initialized above as 'app'
# Vercel will automatically detect and use it