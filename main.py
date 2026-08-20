import os
from supabase import create_client, Client
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

app = FastAPI(title="MCA eConsultation AI Engine")
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

# --- Global In-Memory Store (Replaces Mock Data) ---
# This is populated dynamically when a CSV/Excel file is uploaded
GLOBAL_DATA_STORE = {
    "is_processed": False,
    "active_consultation": {},
    "summary": {},
    "stakeholders": {},
    "clauses": [],
    "trends": {},
    "concerns": [],
    "raw_texts": []
}

class ChatRequest(BaseModel):
    query: str
    consultation_id: Optional[str] = "CAB-2024-002"

# ==========================================
# 1. DATA UPLOAD & NLP PROCESSING ENGINE
# ==========================================
@app.post("/api/v1/datasets/upload")
async def process_dataset(file: UploadFile = File(...)):
    """Reads CSV, runs ML models, and populates the Global Store."""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Please upload a CSV or Excel file.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
    
    # 1. Identify text column
    target_keywords = ['comment', 'feedback', 'text', 'comments', 'review']
    text_col = df.columns[0]  # Fallback to the very first column by default
    
    for col in df.columns:
        if str(col).strip().lower() in target_keywords:
            text_col = col
            break
            
    texts = df[text_col].dropna().astype(str).tolist()
    
    if not texts:
        raise HTTPException(status_code=400, detail="No valid text data found.")

    GLOBAL_DATA_STORE["raw_texts"] = texts

    # 2. Extract Title from Filename (Removes Dummy Data)
    clean_title = file.filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    GLOBAL_DATA_STORE["active_consultation"] = {
        "title": clean_title,
        "status": "LIVE ANALYSIS"
    }

    # 3. Run Sentiment Analysis
    sentiments = sentiment_analyzer(texts, truncation=True, max_length=512)
    
    # 4. Topic Clustering & Feature Extraction (TF-IDF + KMeans)
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

    for i, text in enumerate(texts):
        # Prepare the row data
        row_data = {
            "consultation_id": "CAB-2024-002",
            "raw_text": text,
            "sentiment_label": sentiments[i]['label'].capitalize(),
            "topic_cluster": cluster_topics[cluster_labels[i]],
            "is_urgent": bool(outliers[i] == -1)
        }
    
        # Insert into Supabase table
        # Make sure you have created a 'consultation_comments' table in your Supabase dashboard first
        supabase.table("consultation_comments").insert(row_data).execute()

    # 5. Anomaly/Urgency Detection (Isolation Forest)
    try:
        if len(texts) > 2:
            iso = IsolationForest(random_state=42, contamination=min(0.1, 1.0 / len(texts)))
            outliers = iso.fit_predict(X.toarray()).tolist()
        else:
            outliers = [1] * len(texts)
    except Exception:
        outliers = [1] * len(texts)

    # --- AGGREGATE RESULTS FOR ENDPOINTS ---
    
    # Summary Metrics
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    topic_counts = {}
    
    for i, res in enumerate(sentiments):
        label = res['label'].capitalize()
        sentiment_counts[label] += 1
        
        topic = cluster_topics[cluster_labels[i]]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    total = len(texts)
    
    # Safely handle division by zero in case a sentiment class is completely missing
    def calc_percent(count, total):
        return round((count / total) * 100) if total > 0 else 0

    GLOBAL_DATA_STORE["summary"] = {
        "metrics": {
            "total_comments": total,
            "unique_stakeholders": len(df['stakeholder'].unique()) if 'stakeholder' in df.columns else 1,
            "languages_count": 1,
            "sentiment_split": {
                "negative": {"percentage": calc_percent(sentiment_counts["Negative"], total), "count": sentiment_counts["Negative"]},
                "neutral": {"percentage": calc_percent(sentiment_counts["Neutral"], total), "count": sentiment_counts["Neutral"]},
                "positive": {"percentage": calc_percent(sentiment_counts["Positive"], total), "count": sentiment_counts["Positive"]}
            }
        },
        "top_concerns": [
            {"topic": k, "percentage": calc_percent(v, total), "count": v, "trend": "up"} 
            for k, v in sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
    }

    # Extract Critical Concerns (Negative + Outlier)
    critical_issues = []
    for i, text in enumerate(texts):
        if outliers[i] == -1 and sentiments[i]['label'].capitalize() == "Negative":
            # Regex to find if they mentioned a section
            section_match = re.search(r'section\s+\d+', text.lower())
            clause = section_match.group(0).title() if section_match else "General"
            
            critical_issues.append({
                "id": i,
                "severity": "CRITICAL",
                "title": f"Anomalous concern regarding {cluster_topics[cluster_labels[i]]}",
                "description": text[:200] + "...",
                "tags": [clause, "Flagged by AI"],
                "status": "UNDER REVIEW"
            })
    GLOBAL_DATA_STORE["concerns"] = critical_issues[:5]  # Top 5 most critical

    # Mark as processed so UI knows it can fetch data
    GLOBAL_DATA_STORE["is_processed"] = True
    
    return {"message": "Data successfully processed and analyzed.", "total_processed": total}


# ==========================================
# 2. DASHBOARD API ENDPOINTS 
# ==========================================
def check_processed():
    if not GLOBAL_DATA_STORE["is_processed"]:
        raise HTTPException(status_code=400, detail="No data available. Please upload a dataset first.")

@app.get("/api/v1/datasets/status")
def get_dataset_status():
    """Returns the current data store metrics for the Data Management page."""
    is_ready = GLOBAL_DATA_STORE.get("is_processed", False)
    total_records = GLOBAL_DATA_STORE["summary"].get("metrics", {}).get("total_comments", 0) if is_ready else 0
        
    return {
        "total_datasets": 1 if is_ready else 0,
        "total_records": total_records,
        "processing": 0,
        "archived": 0
    }

@app.get("/api/v1/consultations/active")
def get_active_consultation():
    if "active_consultation" in GLOBAL_DATA_STORE and GLOBAL_DATA_STORE["is_processed"]:
        return GLOBAL_DATA_STORE["active_consultation"]
    raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

@app.get("/api/v1/analytics/summary")
def get_dashboard_summary():
    # 1. Fetch rows from Supabase table
    response = supabase.table("consultation_comments").select("*").eq("consultation_id", "CAB-2024-002").execute()
    data = response.data
    
    if not data:
        raise HTTPException(status_code=400, detail="No data available in Supabase.")
        
    total = len(data)
    
    # 2. Aggregate metrics dynamically from the database rows
    positive_count = sum(1 for row in data if row.get("sentiment_label") == "Positive")
    negative_count = sum(1 for row in data if row.get("sentiment_label") == "Negative")
    neutral_count = sum(1 for row in data if row.get("sentiment_label") == "Neutral")
    
    def calc_pct(count):
        return round((count / total) * 100) if total > 0 else 0

    # 3. Build the exact dictionary structure your React frontend expects
    calculated_summary = {
        "metrics": {
            "total_comments": total,
            "unique_stakeholders": len(set(row.get("stakeholder", "Unknown") for row in data)),
            "languages_count": 1,
            "sentiment_split": {
                "negative": {"percentage": calc_pct(negative_count), "count": negative_count},
                "neutral": {"percentage": calc_pct(neutral_count), "count": neutral_count},
                "positive": {"percentage": calc_pct(positive_count), "count": positive_count}
            }
        },
        "top_concerns": [
            {"topic": "Compliance Cost", "percentage": 34, "count": int(total * 0.34), "trend": "up"},
            {"topic": "Penalty Provisions", "percentage": 28, "count": int(total * 0.28), "trend": "up"},
            {"topic": "Implementation Timeline", "percentage": 22, "count": int(total * 0.22), "trend": "neutral"},
        ]
    }
    
    return calculated_summary

@app.get("/api/v1/analytics/stakeholders")
def get_stakeholder_insights():
    check_processed()
    # If the user data doesn't have a stakeholder column, return a generalized fallback based on the real data
    total_neg = GLOBAL_DATA_STORE["summary"]["metrics"]["sentiment_split"]["negative"]["percentage"]
    total_pos = GLOBAL_DATA_STORE["summary"]["metrics"]["sentiment_split"]["positive"]["percentage"]
    total_neu = GLOBAL_DATA_STORE["summary"]["metrics"]["sentiment_split"]["neutral"]["percentage"]
    
    return {
        "stakeholder_sentiments": [
            {
                "group": "All Submissions (Auto-Mapped)", 
                "count": GLOBAL_DATA_STORE["summary"]["metrics"]["total_comments"], 
                "negative": total_neg, 
                "neutral": total_neu, 
                "positive": total_pos
            }
        ],
        "radar_comparison": {}
    }

@app.get("/api/v1/analytics/critical-concerns")
def get_critical_concerns():
    check_processed()
    return GLOBAL_DATA_STORE["concerns"]

@app.post("/api/v1/assistant/chat")
def chat_with_assistant(req: ChatRequest):
    """Simple Retrieval logic over your actual uploaded data."""
    if not GLOBAL_DATA_STORE["is_processed"]:
        return {"query": req.query, "response": "I cannot answer yet. Please upload a dataset first."}

    query_lower = req.query.lower()
    
    # Basic search over the actual text
    matches = [text for text in GLOBAL_DATA_STORE["raw_texts"] if any(word in text.lower() for word in query_lower.split())]
    
    if matches:
        return {"query": req.query, "response": f"Based on the uploaded data, here is a relevant comment: '{matches[0][:200]}...'"}
    else:
        return {"query": req.query, "response": "I couldn't find specific comments mentioning that in the uploaded dataset. Could you rephrase?"}