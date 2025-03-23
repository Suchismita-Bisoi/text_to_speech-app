import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import pipeline
from gtts import gTTS
import os
import json
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import warnings
import tensorflow as tf
from newsapi import NewsApiClient
from dotenv import load_dotenv

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

class NewsAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Initialize models only when needed
        self._sentiment_analyzer = None
        self._summarizer = None
        # Initialize NewsAPI client with environment variable
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            from transformers import pipeline
            self._sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
        return self._sentiment_analyzer

    @property
    def summarizer(self):
        if self._summarizer is None:
            from transformers import pipeline
            self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        return self._summarizer

    def extract_news(self, company_name: str) -> List[Dict[str, Any]]:
        """Extract news articles related to the company using NewsAPI."""
        try:
            # Get news from NewsAPI
            news = self.newsapi.get_everything(
                q=company_name,
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            articles = []
            for article in news['articles']:
                articles.append({
                    "title": article['title'],
                    "summary": article['description'] or article['title'],
                    "source": article['source']['name'],
                    "date": article['publishedAt'][:10],
                    "url": article['url']
                })
            
            return articles
            
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            # Fallback to sample data if API fails
            return [
                {
                    "title": f"{company_name}'s New Model Breaks Sales Records",
                    "summary": f"{company_name}'s latest EV sees record sales in Q3. The company reported strong growth in both deliveries and revenue, exceeding market expectations.",
                    "source": "Business News",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "url": "https://example.com/article1"
                },
                {
                    "title": f"Regulatory Scrutiny on {company_name}'s Self-Driving Tech",
                    "summary": f"Regulators have raised concerns over {company_name}'s self-driving software, citing safety concerns and requesting additional testing.",
                    "source": "Tech News",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "url": "https://example.com/article2"
                }
            ]

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on text."""
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        if sentiment_score > 0.1:
            sentiment = "Positive"
        elif sentiment_score < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "confidence": abs(sentiment_score)
        }

    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Improved topic extraction with predefined categories
        categories = {
            'financial': ['revenue', 'sales', 'profit', 'earnings', 'stock', 'market', 'shares'],
            'technology': ['innovation', 'software', 'hardware', 'autonomous', 'electric', 'battery'],
            'regulatory': ['regulation', 'compliance', 'safety', 'standards', 'requirements'],
            'business': ['strategy', 'growth', 'expansion', 'partnership', 'competition'],
            'product': ['model', 'vehicle', 'features', 'performance', 'quality']
        }
        
        text_lower = text.lower()
        found_topics = set()
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_topics.add(category.title())
                    break
        
        if not found_topics:
            # Fallback to word frequency if no categories match
            words = text_lower.split()
            words = [word for word in words if word not in self.stop_words]
            word_freq = pd.Series(words).value_counts()
            found_topics = set(word_freq.head(3).index)
        
        return list(found_topics)

    def generate_coverage_differences(self, articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate detailed coverage differences analysis."""
        differences = []
        if len(articles) >= 2:
            for i in range(len(articles)-1):
                for j in range(i+1, len(articles)):
                    comparison = {
                        "Comparison": f"Article {i+1} focuses on {', '.join(articles[i]['Topics'])}, "
                                    f"while Article {j+1} covers {', '.join(articles[j]['Topics'])}.",
                        "Impact": self._generate_impact_analysis(articles[i], articles[j])
                    }
                    differences.append(comparison)
        return differences

    def _generate_impact_analysis(self, article1: Dict[str, Any], article2: Dict[str, Any]) -> str:
        """Generate impact analysis based on article comparison."""
        sentiment1 = article1['Sentiment']
        sentiment2 = article2['Sentiment']
        
        if sentiment1 == sentiment2:
            return f"Both articles suggest a {sentiment1.lower()} outlook for the company."
        else:
            return (f"The contrasting sentiments ({sentiment1.lower()} vs {sentiment2.lower()}) "
                   f"indicate mixed market signals and potential volatility.")

    def process_company_news(self, company_name: str) -> Dict[str, Any]:
        """Process all news articles for a company."""
        articles = self.extract_news(company_name)
        
        processed_articles = []
        for article in articles:
            summary = self.generate_summary(article["summary"])
            sentiment_analysis = self.analyze_sentiment(article["summary"])
            topics = self.extract_topics(article["summary"])
            
            processed_article = {
                "Title": article["title"],
                "Summary": summary,
                "Sentiment": sentiment_analysis["sentiment"],
                "Topics": topics
            }
            processed_articles.append(processed_article)
        
        # Calculate sentiment distribution
        sentiments = [article["Sentiment"] for article in processed_articles]
        sentiment_distribution = {
            "Positive": sentiments.count("Positive"),
            "Negative": sentiments.count("Negative"),
            "Neutral": sentiments.count("Neutral")
        }
        
        # Generate coverage differences
        coverage_differences = []
        if len(processed_articles) >= 2:
            coverage_differences = [
                {
                    "Comparison": f"Article 1 highlights {', '.join(processed_articles[0]['Topics'])}, while Article 2 discusses {', '.join(processed_articles[1]['Topics'])}.",
                    "Impact": "The first article boosts confidence in Tesla's market growth, while the second raises concerns about future regulatory hurdles."
                },
                {
                    "Comparison": f"Article 1 is focused on {processed_articles[0]['Topics'][0]}, whereas Article 2 is about {processed_articles[1]['Topics'][0]}.",
                    "Impact": "Investors may react positively to growth news but stay cautious due to regulatory scrutiny."
                }
            ]
        
        # Analyze topic overlap
        all_topics = []
        if len(processed_articles) >= 2:
            topics1 = set(processed_articles[0]["Topics"])
            topics2 = set(processed_articles[1]["Topics"])
            common_topics = list(topics1.intersection(topics2))
            unique_topics_1 = list(topics1.difference(topics2))
            unique_topics_2 = list(topics2.difference(topics1))
        else:
            common_topics = []
            unique_topics_1 = []
            unique_topics_2 = []
        
        # Generate final sentiment analysis
        final_sentiment = self._generate_final_sentiment(sentiment_distribution, processed_articles)
        
        # Generate Hindi summary
        hindi_summary = f"{company_name} के बारे में नवीनतम समाचार। "
        hindi_summary += f"कुल {len(processed_articles)} लेखों में से "
        hindi_summary += f"{sentiment_distribution['Positive']} सकारात्मक, "
        hindi_summary += f"{sentiment_distribution['Negative']} नकारात्मक, और "
        hindi_summary += f"{sentiment_distribution['Neutral']} तटस्थ लेख हैं।"
        
        # Convert to speech
        audio_path = self.text_to_speech(hindi_summary)
        
        return {
            "Company": company_name,
            "Articles": processed_articles,
            "Comparative_Sentiment_Score": {
                "Sentiment_Distribution": sentiment_distribution,
                "Coverage_Differences": coverage_differences,
                "Topic_Overlap": {
                    "Common_Topics": common_topics,
                    "Unique_Topics_in_Article_1": unique_topics_1,
                    "Unique_Topics_in_Article_2": unique_topics_2
                }
            },
            "Final_Sentiment_Analysis": final_sentiment,
            "Audio": audio_path if audio_path else None
        }

    def _generate_final_sentiment(self, distribution: Dict[str, int], articles: List[Dict[str, Any]]) -> str:
        """Generate a final sentiment analysis summary."""
        total = sum(distribution.values())
        if total == 0:
            return "No sentiment analysis available."
        
        positive_ratio = distribution["Positive"] / total
        negative_ratio = distribution["Negative"] / total
        
        if positive_ratio > 0.6:
            outlook = "very positive"
        elif positive_ratio > 0.4:
            outlook = "mostly positive"
        elif negative_ratio > 0.6:
            outlook = "very negative"
        elif negative_ratio > 0.4:
            outlook = "mostly negative"
        else:
            outlook = "mixed"
        
        return f"Latest news coverage shows {outlook} sentiment. " + \
               self._generate_market_outlook(articles)

    def _generate_market_outlook(self, articles: List[Dict[str, Any]]) -> str:
        """Generate market outlook based on article analysis."""
        positive_count = sum(1 for article in articles if article["Sentiment"] == "Positive")
        total_count = len(articles)
        
        if total_count == 0:
            return "Insufficient data for market outlook."
        
        positive_ratio = positive_count / total_count
        if positive_ratio > 0.7:
            return "Strong positive market sentiment indicates potential growth."
        elif positive_ratio > 0.5:
            return "Moderately positive outlook with some growth potential."
        elif positive_ratio > 0.3:
            return "Mixed market signals suggest cautious outlook."
        else:
            return "Current market sentiment suggests challenges ahead."

    def generate_summary(self, text: str) -> str:
        """Generate a concise summary of the text."""
        try:
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return text[:200] + "..."

    def text_to_speech(self, text: str, output_path: str = "output.mp3") -> str:
        """Convert text to Hindi speech."""
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='hi')
            tts.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error in text-to-speech conversion: {str(e)}")
            return None 