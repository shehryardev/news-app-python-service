import os
import time
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
from googletrans import Translator

# ========== CONFIG ==========
NEWSAPI_KEY = "609deb944282491298eb914e70568779"

# these topics is used as tags for the news articles
TOPICS_GLOBAL = [
    "world news", "global economy", "international relations", "geopolitics", "diplomacy",
    "US politics", "EU", "BRICS", "Russia", "Ukraine", "Israel", "Palestine", "Iran", "India", "China", "Pakistan", "Imran Khan", "PMLN", "PTI", "Tariffs", "politics", "trump"
]

TOPICS_BUSINESS = [
    "startups", "entrepreneurship", "venture capital", "angel investing", "fundraising",
    "IPO", "M&A", "Silicon Valley", "bootstrapping", "small businesses", "SaaS", "B2B", "ecommerce"
]

TOPICS_SPORTS = [
    "sports", "football", "soccer", "basketball", "NBA", "NFL", "cricket", "tennis", "Olympics", "Ronaldo", "Messi", "Real Madrid", "Barcelona", "Manchester United", "Chelsea", "Arsenal", "Liverpool", "Manchester City", "Juventus", "Inter Milan", "Bayern Munich", "PSG", "Real Madrid", "Barcelona", "Manchester United", "Chelsea", "Arsenal", "Liverpool", "Manchester City", "Juventus", "Inter Milan", "Bayern Munich", "PSG"
    "FIFA", "World Cup", "athletics", "golf", "Formula 1", "motorsport", "baseball", "MLB", "NHL", "hockey", "rugby", "boxing", "MMA", "UFC", "cycling", "swimming"
]


TOPICS_AI = [
    "AI", "machine learning", "deep learning", "OpenAI", "ChatGPT", "LLM", "AGI", "AI ethics", "AI regulation",
    "Neural networks", "NLP", "computer vision", "AI in healthcare", "AI in education"
]
TOPICS_DEV = [
    "programming", "coding", "software engineering", "devtools", "open source",
    "frontend", "backend", "React", "Next.js", "JavaScript", "TypeScript", "Python", "GitHub", "Macbook", "Apple"
]

TOPICS_CLOUD = [
    "cloud computing", "AWS", "Azure", "Google Cloud", "serverless", "DevOps", "infrastructure as code", "Docker", "Kubernetes"
]
TOPICS_CRYPTO = [
    "crypto", "blockchain", "Bitcoin", "Ethereum", "Web3", "DeFi", "NFTs", "stablecoins", "smart contracts", "digital currency"
]
TOPICS_SPACE = [
    "space", "space exploration", "rockets", "satellites", "NASA", "astronomy", "Mars", "SpaceX", "Blue Origin", "quantum physics", "quantum computing"
]
TOPICS_ENVIRONMENT = [
    "climate change", "carbon emissions", "sustainability", "green tech", "solar power",
    "batteries", "electric cars", "energy storage", "environment", "drought"
]
TOPICS_HEALTH = [
    "health", "medicine", "mental health", "COVID-19", "vaccines", "public health",
    "biotech", "longevity", "gene editing"
]
TOPICS_EMERGING_TECH = [
    "metaverse", "virtual reality", "augmented reality", "robotics", "automation",
    "brain-computer interfaces", "Musk", "Elon Musk", "technology", "internet"
]
TOPICS_FINANCE = [
    "finance", "economy", "interest rates", "inflation", "banking", "fintech"
]
TOPICS_EDUCATION_WORK = [
    "education", "edtech", "online learning", "remote work", "future of work", "freelancing"
]
TOPICS_MISC = [
    "Amazon", "Google", "Microsoft", "sports", "web development", "Tesla"
]


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "news_db_dev"
COLLECTION_NAME = "news"

# ========== DB SETUP ==========
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
collection.create_index("url", unique=True)

# ========== TRANSLATION SETUP ==========
translator = Translator()

def translate_text_to_english(text, max_retries=3):
    """
    Translate text to English if it's not already in English
    Returns: (translated_text, was_translated, detected_language)
    """
    if not text or not text.strip():
        return text, False, "unknown"
    
    for attempt in range(max_retries):
        try:
            # Detect language first
            detection = translator.detect(text)
            detected_lang = detection.lang
            
            # If already English, return as-is
            if detected_lang == 'en':
                return text, False, detected_lang
            
            # Translate to English
            translation = translator.translate(text, dest='en', src=detected_lang)
            return translation.text, True, detected_lang
            
        except Exception as e:
            print(f"⚠️ Translation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"❌ Translation failed after {max_retries} attempts. Using original text.")
                return text, False, "error"
            time.sleep(1)  # Wait before retry
    
    return text, False, "error"

# ========== FETCH FUNCTION ==========
def fetch_news_for_topic(topic):
    from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={topic}&from={from_date}&sortBy=publishedAt&pageSize=50&apiKey={NEWSAPI_KEY}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            print(f"❌ API error for topic '{topic}': {data}")
            return []

        articles = data.get("articles", [])
        results = []
        for art in articles:
            # Get original text fields
            original_title = art.get("title", "")
            original_description = art.get("description", "")
            original_content = art.get("content", "")
            
            # Translate title
            translated_title, title_was_translated, title_lang = translate_text_to_english(original_title)
            
            # Translate description
            translated_description, desc_was_translated, desc_lang = translate_text_to_english(original_description)
            
            # Translate content (if available)
            translated_content, content_was_translated, content_lang = translate_text_to_english(original_content)
            
            # Determine overall detected language (prioritize title, then description)
            detected_language = title_lang if title_lang != "unknown" else (desc_lang if desc_lang != "unknown" else content_lang)
            was_translated = title_was_translated or desc_was_translated or content_was_translated
            
            article = {
                "tags": [topic],
                "title": translated_title,
                "description": translated_description,
                "content": translated_content,
                "url": art.get("url", ""),
                "image": art.get("urlToImage", ""),
                "source": art.get("source", {}).get("name", ""),
                "publishedAt": art.get("publishedAt", ""),
                "fetchedAt": datetime.utcnow(),
                # Translation metadata
                "originalLanguage": detected_language,
                "wasTranslated": was_translated,
                "originalTitle": original_title if title_was_translated else None,
                "originalDescription": original_description if desc_was_translated else None,
                "originalContent": original_content if content_was_translated else None
            }
            results.append(article)
        return results

    except Exception as e:
        print(f"⚠️ Error fetching news for topic '{topic}': {e}")
        return []

# ========== MAIN ==========
def populate_news():
    # List of all topic lists
    all_topic_lists = [
                ("CLOUD", TOPICS_CLOUD),
        ("CRYPTO", TOPICS_CRYPTO),
        ("SPACE", TOPICS_SPACE),
        ("ENVIRONMENT", TOPICS_ENVIRONMENT),
        ("HEALTH", TOPICS_HEALTH),
        ("EMERGING_TECH", TOPICS_EMERGING_TECH),
        ("FINANCE", TOPICS_FINANCE),
        ("EDUCATION_WORK", TOPICS_EDUCATION_WORK),
        ("MISC", TOPICS_MISC),
        ("GLOBAL", TOPICS_GLOBAL),
        ("BUSINESS", TOPICS_BUSINESS),
        ("SPORTS", TOPICS_SPORTS),
        ("AI", TOPICS_AI),
        ("DEV", TOPICS_DEV)
    ]
    
    for category_name, topic_list in all_topic_lists:
        print(f"\n📂 Processing category: {category_name}")
        print(f"📊 Total topics in this category: {len(topic_list)}")
        
        for topic in topic_list:
            print(f"\n🔍 Fetching news for topic: {topic}")
            articles = fetch_news_for_topic(topic)
            print(f"→ Fetched {len(articles)} articles")

            inserted = 0
            translated_count = 0
            for article in articles:
                try:
                    existing = collection.find_one({"url": article["url"]})
                    if existing:
                        # merge tags if same article already exists
                        new_tags = list(set(existing.get("tags", []) + article["tags"]))
                        collection.update_one({"url": article["url"]}, {"$set": {"tags": new_tags}})
                    else:
                        collection.insert_one(article)
                        inserted += 1
                        if article.get("wasTranslated", False):
                            translated_count += 1
                            print(f"  📝 Translated from {article.get('originalLanguage', 'unknown')}: {article['title'][:50]}...")
                except Exception as e:
                    print("❌ Insert error:", e)
            
            print(f"✅ Inserted {inserted} new articles for topic: {topic}")
            if translated_count > 0:
                print(f"🌐 Translated {translated_count} articles to English")
            time.sleep(1)  # small delay between topics to avoid API rate limiting
        
        print(f"🎉 Completed category: {category_name}")
        print(f"⏱️  Waiting 5 seconds before next category...")
        time.sleep(5)  # 5 second gap between topic lists

if __name__ == "__main__":
    populate_news()
