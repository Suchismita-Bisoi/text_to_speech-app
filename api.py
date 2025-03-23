from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from utils import NewsAnalyzer
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="News Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NewsAnalyzer
news_analyzer = NewsAnalyzer()

class Article(BaseModel):
    Title: str
    Summary: str
    Sentiment: str
    Topics: List[str]

class TopicOverlap(BaseModel):
    Common_Topics: List[str]
    Unique_Topics_in_Article_1: List[str]
    Unique_Topics_in_Article_2: List[str]

class CoverageDifference(BaseModel):
    Comparison: str
    Impact: str

class ComparativeSentimentScore(BaseModel):
    Sentiment_Distribution: Dict[str, int]
    Coverage_Differences: List[CoverageDifference]
    Topic_Overlap: TopicOverlap

class AnalysisResponse(BaseModel):
    Company: str
    Articles: List[Article]
    Comparative_Sentiment_Score: ComparativeSentimentScore
    Final_Sentiment_Analysis: str
    Audio: str

@app.get("/")
async def root():
    return {"message": "Welcome to the News Analysis API"}

@app.get("/api/news/{company_name}", response_model=AnalysisResponse)
async def analyze_company_news(company_name: str):
    try:
        result = news_analyzer.process_company_news(company_name)
        # Transform the result to match the expected format
        formatted_result = {
            "Company": company_name,
            "Articles": [
                {
                    "Title": article["Title"],
                    "Summary": article["Summary"],
                    "Sentiment": article["Sentiment"],
                    "Topics": article["Topics"]
                } for article in result["Articles"]
            ],
            "Comparative_Sentiment_Score": {
                "Sentiment_Distribution": result["Comparative_Sentiment_Score"]["Sentiment_Distribution"],
                "Coverage_Differences": result["Comparative_Sentiment_Score"]["Coverage_Differences"],
                "Topic_Overlap": result["Comparative_Sentiment_Score"]["Topic_Overlap"]
            },
            "Final_Sentiment_Analysis": result["Final_Sentiment_Analysis"],
            "Audio": result["Audio"]
        }
        return formatted_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def text_to_speech(text: str):
    try:
        audio_path = news_analyzer.text_to_speech(text)
        return {"audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 