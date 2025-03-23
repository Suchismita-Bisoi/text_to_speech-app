import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configure the page
st.set_page_config(
    page_title="News Analysis Dashboard",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .article-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 1rem;
        background-color: white;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: gray;
        font-weight: bold;
    }
    .topic-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 15px;
        background-color: #f0f2f6;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📰 News Analysis Dashboard")
st.markdown("""
    This dashboard provides comprehensive analysis of news articles related to companies,
    including sentiment analysis, topic extraction, and Hindi text-to-speech summaries.
""")

# Sidebar
st.sidebar.title("Settings")
company_name = st.sidebar.text_input("Enter Company Name", "Tesla")

# Main content
if st.sidebar.button("Analyze News"):
    with st.spinner("Analyzing news articles..."):
        try:
            # Call the API
            response = requests.get(f"http://localhost:8000/api/news/{company_name}")
            data = response.json()
            
            # Display company name
            st.header(f"Analysis for {data['Company']}")
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display sentiment distribution
                st.subheader("Sentiment Distribution")
                sentiment_df = pd.DataFrame(
                    list(data['Comparative_Sentiment_Score']['Sentiment_Distribution'].items()),
                    columns=['Sentiment', 'Count']
                )
                fig = px.pie(sentiment_df, values='Count', names='Sentiment',
                           color='Sentiment',
                           color_discrete_map={'Positive': 'green',
                                             'Negative': 'red',
                                             'Neutral': 'gray'})
                st.plotly_chart(fig)
            
            with col2:
                # Display topic overlap
                st.subheader("Topic Analysis")
                topic_overlap = data['Comparative_Sentiment_Score']['Topic_Overlap']
                
                st.markdown("**Common Topics:**")
                for topic in topic_overlap['Common_Topics']:
                    st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
                
                st.markdown("**Unique Topics in Article 1:**")
                for topic in topic_overlap['Unique_Topics_in_Article_1']:
                    st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
                
                st.markdown("**Unique Topics in Article 2:**")
                for topic in topic_overlap['Unique_Topics_in_Article_2']:
                    st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
            
            # Display coverage differences
            st.subheader("Coverage Analysis")
            for diff in data['Comparative_Sentiment_Score']['Coverage_Differences']:
                with st.expander("View Comparison"):
                    st.markdown(f"**Comparison:** {diff['Comparison']}")
                    st.markdown(f"**Impact:** {diff['Impact']}")
            
            # Display final sentiment analysis
            st.subheader("Market Outlook")
            st.info(data['Final_Sentiment_Analysis'])
            
            # Display articles
            st.subheader("Articles")
            for article in data['Articles']:
                with st.container():
                    st.markdown("""
                        <div class="article-card">
                    """, unsafe_allow_html=True)
                    
                    # Article title and sentiment
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {article['Title']}")
                    with col2:
                        sentiment_class = f"sentiment-{article['Sentiment'].lower()}"
                        st.markdown(f'<div class="{sentiment_class}">{article["Sentiment"]}</div>',
                                  unsafe_allow_html=True)
                    
                    # Article summary
                    st.markdown(article['Summary'])
                    
                    # Topics
                    st.markdown("**Topics:**")
                    for topic in article['Topics']:
                        st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Audio player
            if data['Audio']:
                st.subheader("Hindi Summary")
                try:
                    st.audio(data['Audio'])
                except Exception as e:
                    st.error(f"Error playing audio: {str(e)}")
                    st.markdown("Audio playback is not available. Please try again later.")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Built with ❤️ using Streamlit, FastAPI, and Transformers</p>
        <p>© 2024 News Analysis Dashboard</p>
    </div>
""", unsafe_allow_html=True) 