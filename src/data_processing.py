"""
Data processing utilities for stock and Twitter data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """Main class for processing stock and Twitter data"""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use absolute path to the project root
            current_dir = os.getcwd()
            # Navigate up to project root if we're in notebooks folder
            if os.path.basename(current_dir) == 'notebooks':
                data_dir = os.path.join(os.path.dirname(current_dir), "data")
            else:
                data_dir = os.path.join(current_dir, "data")
        
        self.data_dir = data_dir
        self.stock_data_dir = os.path.join(data_dir, "stock_data")
        self.twitter_data_dir = os.path.join(data_dir, "twitter_data_new")
        
        # Debug: Print the actual paths being used
        print(f"Project root: {os.getcwd()}")
        print(f"Data directory: {self.data_dir}")
        print(f"Stock data directory: {self.stock_data_dir}")
        print(f"Twitter data directory: {self.twitter_data_dir}")
        
        # Verify directories exist
        if not os.path.exists(self.stock_data_dir):
            raise FileNotFoundError(f"Stock data directory not found: {self.stock_data_dir}")
        if not os.path.exists(self.twitter_data_dir):
            raise FileNotFoundError(f"Twitter data directory not found: {self.twitter_data_dir}")
        
        # List files in each directory
        print(f"\nStock data files: {os.listdir(self.stock_data_dir)}")
        print(f"Twitter data files: {os.listdir(self.twitter_data_dir)}")
    
    def test_paths(self):
        """Test if all paths are working correctly"""
        print("\n=== TESTING PATHS ===")
        print(f"Stock data dir exists: {os.path.exists(self.stock_data_dir)}")
        print(f"Twitter data dir exists: {os.path.exists(self.twitter_data_dir)}")
        
        if os.path.exists(self.stock_data_dir):
            stock_files = [f for f in os.listdir(self.stock_data_dir) if f.endswith('.csv')]
            print(f"Stock files found: {len(stock_files)}")
            print(f"Stock files: {stock_files}")
        
        if os.path.exists(self.twitter_data_dir):
            twitter_files = [f for f in os.listdir(self.twitter_data_dir) if f.endswith('.csv')]
            print(f"Twitter files found: {len(twitter_files)}")
            print(f"Twitter files: {twitter_files}")
        
        return True
    
    def discover_and_process_all_stocks(self):
        """Discover all available stocks and Twitter data, then process them"""
        print("=== DISCOVERING AVAILABLE DATA ===")
        
        # Get stock files
        try:
            stock_files = [f for f in os.listdir(self.stock_data_dir) if f.endswith('.csv')]
            stock_symbols = [f.replace('.csv', '') for f in stock_files]
            print(f"✅ Found {len(stock_symbols)} stock files: {stock_symbols}")
        except Exception as e:
            print(f"❌ Error reading stock directory: {e}")
            stock_symbols = []
        
        # Get Twitter files
        try:
            twitter_files = [f for f in os.listdir(self.twitter_data_dir) if f.endswith('.csv')]
            twitter_symbols = [f.replace('.csv', '') for f in twitter_files]
            print(f"✅ Found {len(twitter_symbols)} Twitter files: {twitter_symbols}")
        except Exception as e:
            print(f"❌ Error reading Twitter directory: {e}")
            twitter_symbols = []
        
        # Find intersection
        if stock_symbols and twitter_symbols:
            complete_stocks = list(set(stock_symbols) & set(twitter_symbols))
            print(f"\n🎯 Stocks with complete data: {complete_stocks}")
            return complete_stocks
        else:
            print("❌ No complete data found")
            return []
    
    def get_available_stocks(self) -> List[str]:
        """Get list of available stock symbols"""
        try:
            stock_files = [f.replace('.csv', '') for f in os.listdir(self.stock_data_dir) 
                          if f.endswith('.csv')]
            return stock_files
        except Exception as e:
            print(f"Error reading stock directory: {e}")
            return []
    
    def get_available_twitter_data(self) -> List[str]:
        """Get list of available Twitter data for stocks"""
        try:
            twitter_files = [f.replace('.csv', '') for f in os.listdir(self.twitter_data_dir) 
                            if f.endswith('.csv')]
            return twitter_files
        except Exception as e:
            print(f"Error reading Twitter directory: {e}")
            return []
    
    def get_stocks_with_complete_data(self) -> List[str]:
        """Get list of stocks that have both stock and Twitter data"""
        stock_symbols = self.get_available_stocks()
        twitter_symbols = self.get_available_twitter_data()
        
        # Find intersection (stocks with both data types)
        complete_stocks = list(set(stock_symbols) & set(twitter_symbols))
        print(f"Found {len(complete_stocks)} stocks with complete data: {complete_stocks}")
        return complete_stocks
    
    def load_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """Load stock data for a given symbol"""
        file_path = os.path.join(self.stock_data_dir, f"{stock_symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Check if 'Datetime' column exists (stock data format)
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
            elif 'Date' in df.columns:
                # Fallback for 'Date' column
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                raise ValueError(f"No 'Datetime' or 'Date' column found in {stock_symbol}.csv")
            return df
        else:
            raise FileNotFoundError(f"Stock data file not found: {file_path}")
    
    def load_twitter_data(self, stock_symbol: str, use_new: bool = True) -> pd.DataFrame:
        """Load Twitter data for a given stock symbol"""
        # Always use twitter_data_new since that's where all the data is now
        data_dir = self.twitter_data_dir
        
        file_path = os.path.join(data_dir, f"{stock_symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            return df
        else:
            raise FileNotFoundError(f"Twitter data file not found: {file_path}")
    
    def merge_stock_twitter_data(self, stock_symbol: str, 
                                days_lookback: int = 7) -> pd.DataFrame:
        """Merge stock data with Twitter sentiment data"""
        # Load data
        stock_df = self.load_stock_data(stock_symbol)
        twitter_df = self.load_twitter_data(stock_symbol)
        
        # Aggregate Twitter data by date
        twitter_daily = twitter_df.groupby(twitter_df['Datetime'].dt.date).agg({
            'Tweet': 'count',
            'Likes': 'sum',
            'Retweets': 'sum',
            'Replies': 'sum',
            'Quotes': 'sum'
        }).rename(columns={'Tweet': 'tweet_count'})
        
        # Add sentiment columns if they exist
        if 'Positive' in twitter_df.columns:
            sentiment_cols = ['Positive', 'Neutral', 'Negative']
            for col in sentiment_cols:
                if col in twitter_df.columns:
                    twitter_daily[f'{col}_mean'] = twitter_df.groupby(twitter_df['Datetime'].dt.date)[col].mean()
                    twitter_daily[f'{col}_std'] = twitter_df.groupby(twitter_df['Datetime'].dt.date)[col].std()
        
        twitter_daily.index = pd.to_datetime(twitter_daily.index)
        
        # Merge with stock data
        merged_df = stock_df.merge(twitter_daily, 
                                 left_index=True, 
                                 right_index=True, 
                                 how='left')
        
        # Forward fill missing Twitter data
        merged_df = merged_df.fillna(method='ffill')
        
        # Add rolling averages for Twitter metrics
        for col in ['tweet_count', 'Likes', 'Retweets', 'Replies', 'Quotes']:
            if col in merged_df.columns:
                merged_df[f'{col}_7d_avg'] = merged_df[col].rolling(window=7).mean()
                merged_df[f'{col}_30d_avg'] = merged_df[col].rolling(window=30).mean()
        
        return merged_df
    
    def clean_twitter_text(self, text: str) -> str:
        """Clean Twitter text data"""
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def prepare_dataset(self, stock_symbol: str, 
                       target_column: str = 'Close',
                       prediction_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare dataset for machine learning"""
        # Get merged data
        df = self.merge_stock_twitter_data(stock_symbol)
        
        # Create target variable (future price)
        df[f'{target_column}_future'] = df[target_column].shift(-prediction_days)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        feature_columns = [col for col in df.columns 
                         if col not in [f'{target_column}_future', 'Date']]
        
        X = df[feature_columns]
        y = df[f'{target_column}_future']
        
        return X, y
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Extract date components
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Create cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, 
                           stock_symbol: str, 
                           filename: str) -> None:
        """Save processed data to file"""
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f"{stock_symbol}_{filename}.csv")
        df.to_csv(file_path)
        print(f"Processed data saved to: {file_path}")

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Get available stocks
    stocks = processor.get_available_stocks()
    print(f"Available stocks: {stocks}")
    
    # Process a sample stock
    if stocks:
        sample_stock = stocks[0]
        print(f"\nProcessing {sample_stock}...")
        
        try:
            X, y = processor.prepare_dataset(sample_stock)
            print(f"Dataset shape: X={X.shape}, y={y.shape}")
            print(f"Features: {list(X.columns)}")
        except Exception as e:
            print(f"Error processing {sample_stock}: {e}")
