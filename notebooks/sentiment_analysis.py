"""
Sentiment analysis for Twitter data using multiple approaches
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Multi-method sentiment analyzer for Twitter data"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Custom VADER lexicon for financial terms
        self.financial_lexicon = {
            'bull': 2.0, 'bear': -2.0, 'rally': 2.0, 'crash': -3.0,
            'surge': 2.5, 'plunge': -2.5, 'soar': 2.0, 'tumble': -2.0,
            'gain': 1.5, 'loss': -1.5, 'profit': 2.0, 'loss': -1.5,
            'growth': 1.5, 'decline': -1.5, 'positive': 1.5, 'negative': -1.5,
            'strong': 1.0, 'weak': -1.0, 'up': 1.0, 'down': -1.0,
            'high': 1.0, 'low': -1.0, 'good': 1.0, 'bad': -1.0,
            'excellent': 2.0, 'terrible': -2.0, 'amazing': 2.0, 'awful': -2.0
        }
        
        # Update VADER lexicon
        self.vader_analyzer.lexicon.update(self.financial_lexicon)
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        
        return text.strip()
    
    def get_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using TextBlob"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {'polarity': 0.0, 'subjectivity': 0.0}
            
            blob = TextBlob(cleaned_text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using VADER"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
            
            scores = self.vader_analyzer.polarity_scores(cleaned_text)
            return scores
        except Exception as e:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
    
    def get_combined_sentiment(self, text: str) -> Dict[str, float]:
        """Get combined sentiment scores from multiple methods"""
        textblob_scores = self.get_textblob_sentiment(text)
        vader_scores = self.get_vader_sentiment(text)
        
        # Combine scores (weighted average)
        combined_polarity = (textblob_scores['polarity'] + vader_scores['compound']) / 2
        
        return {
            'textblob_polarity': textblob_scores['polarity'],
            'textblob_subjectivity': textblob_scores['subjectivity'],
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'combined_polarity': combined_polarity
        }
    
    def analyze_tweets_batch(self, tweets_df: pd.DataFrame, 
                           text_column: str = 'Tweet') -> pd.DataFrame:
        """Analyze sentiment for a batch of tweets"""
        df = tweets_df.copy()
        
        # Initialize sentiment columns
        sentiment_columns = [
            'textblob_polarity', 'textblob_subjectivity',
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'combined_polarity'
        ]
        
        for col in sentiment_columns:
            df[col] = 0.0
        
        # Analyze sentiment for each tweet
        print("Analyzing sentiment for tweets...")
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx} tweets...")
            
            text = row[text_column]
            sentiment_scores = self.get_combined_sentiment(text)
            
            for col, score in sentiment_scores.items():
                df.at[idx, col] = score
        
        return df
    
    def get_daily_sentiment_summary(self, tweets_df: pd.DataFrame, 
                                  date_column: str = 'Datetime') -> pd.DataFrame:
        """Get daily sentiment summary from tweets"""
        df = tweets_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df['date'] = df[date_column].dt.date
        
        # Group by date and calculate sentiment metrics
        daily_sentiment = df.groupby('date').agg({
            'textblob_polarity': ['mean', 'std', 'count'],
            'textblob_subjectivity': ['mean', 'std'],
            'vader_compound': ['mean', 'std'],
            'vader_positive': ['mean', 'std'],
            'vader_negative': ['mean', 'std'],
            'vader_neutral': ['mean', 'std'],
            'combined_polarity': ['mean', 'std'],
            'Likes': 'sum',
            'Retweets': 'sum',
            'Replies': 'sum',
            'Quotes': 'sum'
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
        
        # Reset index
        daily_sentiment.reset_index(inplace=True)
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        return daily_sentiment
    
    def create_sentiment_features(self, daily_sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional sentiment features for ML"""
        df = daily_sentiment_df.copy()
        
        # Sentiment momentum (change in sentiment over time)
        df['sentiment_momentum'] = df['combined_polarity_mean'].diff()
        df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
        
        # Sentiment volatility
        df['sentiment_volatility'] = df['combined_polarity_mean'].rolling(window=7).std()
        
        # Sentiment trends
        df['sentiment_trend_7d'] = df['combined_polarity_mean'].rolling(window=7).mean()
        df['sentiment_trend_30d'] = df['combined_polarity_mean'].rolling(window=30).mean()
        
        # Engagement-weighted sentiment
        df['engagement'] = df['Likes_sum'] + df['Retweets_sum'] + df['Replies_sum']
        df['weighted_sentiment'] = (df['combined_polarity_mean'] * df['engagement']) / df['engagement'].max()
        
        # Sentiment extremes
        df['sentiment_extreme_positive'] = (df['combined_polarity_mean'] > 0.5).astype(int)
        df['sentiment_extreme_negative'] = (df['combined_polarity_mean'] < -0.5).astype(int)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def get_sentiment_insights(self, daily_sentiment_df: pd.DataFrame) -> Dict:
        """Get insights from sentiment analysis"""
        df = daily_sentiment_df.copy()
        
        insights = {
            'total_days': len(df),
            'avg_sentiment': df['combined_polarity_mean'].mean(),
            'sentiment_std': df['combined_polarity_mean'].std(),
            'most_positive_day': df.loc[df['combined_polarity_mean'].idxmax(), 'date'],
            'most_negative_day': df.loc[df['combined_polarity_mean'].idxmin(), 'date'],
            'positive_days_ratio': (df['combined_polarity_mean'] > 0).mean(),
            'negative_days_ratio': (df['combined_polarity_mean'] < 0).mean(),
            'neutral_days_ratio': (df['combined_polarity_mean'] == 0).mean(),
            'avg_engagement': df['engagement'].mean() if 'engagement' in df.columns else 0
        }
        
        return insights

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test with sample text
    sample_texts = [
        "This stock is going to the moon! 🚀",
        "Terrible earnings report, stock will crash",
        "Market is stable today, no major changes",
        "Great news for investors, strong growth ahead"
    ]
    
    print("Testing sentiment analysis:")
    for text in sample_texts:
        sentiment = analyzer.get_combined_sentiment(text)
        print(f"\nText: {text}")
        print(f"Combined Polarity: {sentiment['combined_polarity']:.3f}")
        print(f"VADER Compound: {sentiment['vader_compound']:.3f}")
        print(f"TextBlob Polarity: {sentiment['textblob_polarity']:.3f}")
