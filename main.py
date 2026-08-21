import os
import io
import re
from typing import Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from supabase import create_client, Client

app = FastAPI(title="MCA eConsultation AI Engine")

# --- Supabase Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://typdodfsvrvhyxaumcqt.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_OUinriqiXgfizEsGkBsDNg_JEOdy2pi")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NLP Models Setup ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

# --- Global State Tracking ---
# Used only to track the most recently uploaded file ID for fallbacks
GLOBAL_DATA_STORE = {
    "active_consultation_id": None
}

class ChatRequest(BaseModel):
    query: str
    consultation_id: Optional[str] = None


# ==========================================
# 1. DATA UPLOAD & NLP PROCESSING ENGINE
# ==========================================
@app.post("/api/v1/datasets/upload")
async def process_dataset(file: UploadFile = File(...)):
    """Reads CSV, runs ML models, and populates Supabase with a Dynamic ID."""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Please upload a CSV or Excel file.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
    
    target_keywords = ['comment', 'feedback', 'text', 'comments', 'review']
    text_col = df.columns[0]
    
    for col in df.columns:
        if str(col).strip().lower() in target_keywords:
            text_col = col
            break
            
    texts = df[text_col].dropna().astype(str).tolist()
    
    if not texts:
        raise HTTPException(status_code=400, detail="No valid text data found.")

    # Generate Dynamic ID from Filename (e.g., "Sample1.csv" -> "SAMPLE1")
    raw_name = file.filename.rsplit('.', 1)[0]
    consultation_id = raw_name.replace(' ', '_').upper()
    clean_title = raw_name.replace('_', ' ').replace('-', ' ').title()
    
    GLOBAL_DATA_STORE["active_consultation_id"] = consultation_id

    # Run AI Models
    sentiments = sentiment_analyzer(texts, truncation=True, max_length=512)
    
    n_clusters = min(5, len(texts))
    tfidf = TfidfVectorizer(stop_words="english", max_features=100, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X).tolist()
    
    feature_names = np.array(tfidf.get_feature_names_out())
    cluster_topics = {}
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_idx = center.argsort()[::-1][:1]
        cluster_topics[i] = feature_names[top_idx][0].title() if len(feature_names) > 0 else f"Topic {i+1}"

    outliers = [1] * len(texts) 
    try:
        if len(texts) > 2:
            # Make the AI more sensitive for small test datasets (flags up to 20%)
            contamination_rate = 0.25 if len(texts) < 50 else 0.05
            iso = IsolationForest(random_state=42, contamination=contamination_rate)
            outliers = iso.fit_predict(X.toarray()).tolist()
    except Exception as e:
        print(f"Warning: Anomaly detection skipped due to error: {e}")

    # Database Insertion
    for i, text in enumerate(texts):
        row_data = {
            "consultation_id": consultation_id,
            "raw_text": text,
            "sentiment_label": sentiments[i]['label'].capitalize(),
            "topic_cluster": cluster_topics[cluster_labels[i]],
            "is_urgent": bool(outliers[i] == -1)
        }
        
        try:
            supabase.table("consultation_comments").insert(row_data).execute()
        except Exception as e:
            print(f"Supabase error on row {i}: {e}")

    return {
        "message": "Data successfully processed and analyzed.", 
        "total_processed": len(texts),
        "consultation_id": consultation_id,
        "title": clean_title
    }


# ==========================================
# 2. DASHBOARD API ENDPOINTS 
# ==========================================

@app.get("/api/v1/consultations/list")
def get_consultations_list():
    """Fetches a list of all unique datasets currently saved in Supabase."""
    response = supabase.table("consultation_comments").select("consultation_id").execute()
    if not response.data:
        return []
        
    unique_ids = list(set(row["consultation_id"] for row in response.data if row.get("consultation_id")))
    
    dropdown_options = []
    for cid in unique_ids:
        title = cid.replace('_', ' ').title()
        dropdown_options.append({"id": cid, "title": title})
        
    return dropdown_options


@app.get("/api/v1/consultations/active")
def get_active_consultation(consultation_id: Optional[str] = None):
    target_id = consultation_id or GLOBAL_DATA_STORE["active_consultation_id"]
    if not target_id:
        raise HTTPException(status_code=400, detail="No active dataset.")
        
    return {
        "id": target_id,
        "title": target_id.replace('_', ' ').title(),
        "status": "LIVE ANALYSIS"
    }


@app.get("/api/v1/analytics/summary")
def get_dashboard_summary(consultation_id: Optional[str] = None):
    """Fetches summary metrics strictly for the requested dataset ID."""
    target_id = consultation_id or GLOBAL_DATA_STORE["active_consultation_id"]
    if not target_id:
        raise HTTPException(status_code=400, detail="No dataset ID provided.")

    response = supabase.table("consultation_comments").select("*").eq("consultation_id", target_id).execute()
    data = response.data
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for this consultation.")
        
    total = len(data)
    positive_count = sum(1 for row in data if row.get("sentiment_label") == "Positive")
    negative_count = sum(1 for row in data if row.get("sentiment_label") == "Negative")
    neutral_count = sum(1 for row in data if row.get("sentiment_label") == "Neutral")
    
    def calc_pct(count):
        return round((count / total) * 100) if total > 0 else 0

    topic_tally = {}
    for row in data:
        topic = row.get("topic_cluster", "Unknown")
        topic_tally[topic] = topic_tally.get(topic, 0) + 1
        
    top_concerns = [
        {"topic": k, "percentage": calc_pct(v), "count": v, "trend": "up"} 
        for k, v in sorted(topic_tally.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    return {
        "metrics": {
            "total_comments": total,
            "unique_stakeholders": len(set(row.get("stakeholder", "Unknown") for row in data if row.get("stakeholder"))),
            "languages_count": 1,
            "sentiment_split": {
                "negative": {"percentage": calc_pct(negative_count), "count": negative_count},
                "neutral": {"percentage": calc_pct(neutral_count), "count": neutral_count},
                "positive": {"percentage": calc_pct(positive_count), "count": positive_count}
            }
        },
        "top_concerns": top_concerns
    }


@app.get("/api/v1/analytics/critical-concerns")
def get_critical_concerns(consultation_id: Optional[str] = None):
    target_id = consultation_id or GLOBAL_DATA_STORE["active_consultation_id"]
    if not target_id:
        raise HTTPException(status_code=400, detail="No dataset ID provided.")
        
    # Query Supabase specifically for urgent/negative comments attached to this dataset ID
    response = supabase.table("consultation_comments").select("*").eq("consultation_id", target_id).eq("is_urgent", True).eq("sentiment_label", "Negative").execute()
    data = response.data
    
    critical_issues = []
    for i, row in enumerate(data):
        text = row.get("raw_text", "")
        topic = row.get("topic_cluster", "Unknown")
        section_match = re.search(r'section\s+\d+', text.lower())
        clause = section_match.group(0).title() if section_match else "General"
        
        critical_issues.append({
            "id": row.get("id", i),
            "severity": "CRITICAL",
            "title": f"Anomalous concern regarding {topic}",
            "description": text[:200] + "..." if len(text) > 200 else text,
            "tags": [clause, "Flagged by AI"],
            "status": "UNDER REVIEW"
        })
        
    return critical_issues[:5]


@app.get("/api/v1/analytics/stakeholders")
def get_stakeholder_insights(consultation_id: Optional[str] = None):
    target_id = consultation_id or GLOBAL_DATA_STORE["active_consultation_id"]
    if not target_id:
        return {"stakeholder_sentiments": [], "radar_comparison": {}}

    response = supabase.table("consultation_comments").select("*").eq("consultation_id", target_id).execute()
    data = response.data
    total = len(data)
    
    if total == 0:
        return {"stakeholder_sentiments": [], "radar_comparison": {}}
        
    pos = sum(1 for row in data if row.get("sentiment_label") == "Positive")
    neg = sum(1 for row in data if row.get("sentiment_label") == "Negative")
    neu = sum(1 for row in data if row.get("sentiment_label") == "Neutral")
    
    def calc_pct(c): return round((c / total) * 100) if total > 0 else 0

    return {
        "stakeholder_sentiments": [
            {
                "group": "All Submissions (Auto-Mapped)", 
                "count": total, 
                "negative": calc_pct(neg), 
                "neutral": calc_pct(neu), 
                "positive": calc_pct(pos)
            }
        ],
        "radar_comparison": {}
    }


@app.get("/api/v1/datasets/status")
def get_dataset_status():
    """Counts unique datasets and total records across the entire database."""
    response = supabase.table("consultation_comments").select("consultation_id").execute()
    unique_ids = len(set(row["consultation_id"] for row in response.data if row.get("consultation_id")))
    total_records = len(response.data) if response.data else 0
        
    return {
        "total_datasets": unique_ids,
        "total_records": total_records,
        "processing": 0,
        "archived": 0
    }


@app.post("/api/v1/assistant/chat")
def chat_with_assistant(req: ChatRequest):
    target_id = req.consultation_id or GLOBAL_DATA_STORE["active_consultation_id"]
    if not target_id:
        return {"query": req.query, "response": "Please select or upload a dataset first."}

    query_lower = req.query.lower().strip()
    
    # 1. Handle Basic Greetings
    if query_lower in ["hi", "hello", "hey", "help", "who are you"]:
        return {
            "query": req.query, 
            "response": "Hello! I can help you search this dataset. Try asking me for 'negative comments', 'positive feedback', 'top concerns', or ask about a specific topic like 'compliance'."
        }

    # Fetch full dataset to access rows
    response = supabase.table("consultation_comments").select("*").eq("consultation_id", target_id).execute()
    data = response.data
    
    if not data:
        return {"query": req.query, "response": "This dataset appears to be empty."}

    # 2. Handle Sentiment-Specific Queries
    if "negative" in query_lower:
        negatives = [row["raw_text"] for row in data if row.get("sentiment_label") == "Negative"]
        if negatives:
            return {"query": req.query, "response": f"I found {len(negatives)} negative comments. Here is a sample:\n\n'{negatives[0]}'"}
        return {"query": req.query, "response": "There are no negative comments recorded in this dataset."}
        
    if "positive" in query_lower:
        positives = [row["raw_text"] for row in data if row.get("sentiment_label") == "Positive"]
        if positives:
            return {"query": req.query, "response": f"I found {len(positives)} positive comments. Here is a sample:\n\n'{positives[0]}'"}
        return {"query": req.query, "response": "There are no positive comments recorded in this dataset."}

    # 3. Handle Analytical Queries ("top concerns", "issues", "summary")
    if any(term in query_lower for term in ["concern", "concerns", "issue", "issues", "top", "summarize", "summary"]):
        topic_tally = {}
        for row in data:
            topic = row.get("topic_cluster", "General")
            topic_tally[topic] = topic_tally.get(topic, 0) + 1
        
        sorted_topics = sorted(topic_tally.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_str = ", ".join([f"**{t[0]}** ({t[1]} mentions)" for t in sorted_topics])
        return {
            "query": req.query,
            "response": f"Based on the analysis of this dataset, the primary discussion themes and top concerns are: {topic_str}."
        }

    # 4. Smart Keyword Search (Filter out filler words)
    stop_words = {"give", "all", "show", "me", "what", "are", "the", "a", "an", "is", "comments", "comment", "feedback", "about", "raised", "by", "stakeholders"}
    query_words = [w for w in query_lower.split() if w not in stop_words]
    
    if not query_words:
        return {"query": req.query, "response": "Could you please be more specific about what you are looking for?"}
        
    matches = [row["raw_text"] for row in data if any(qw in str(row.get("raw_text", "")).lower() for qw in query_words)]
    
    if matches:
        return {"query": req.query, "response": f"I found {len(matches)} comments mentioning those terms. Here is the most relevant one:\n\n'{matches[0][:300]}...'"}
    else:
        return {"query": req.query, "response": f"I couldn't find any specific comments mentioning '{' '.join(query_words)}' in this dataset."}