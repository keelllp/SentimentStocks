import os
import sys

# Add the project root to Python path
project_root = os.getcwd()
sys.path.append(project_root)

print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# Check if data directory exists
data_dir = os.path.join(project_root, "data")
print(f"Data directory: {data_dir}")
print(f"Data directory exists: {os.path.exists(data_dir)}")

if os.path.exists(data_dir):
    print(f"Contents of data directory: {os.listdir(data_dir)}")
    
    # Check stock_data
    stock_data_dir = os.path.join(data_dir, "stock_data")
    print(f"\nStock data directory: {stock_data_dir}")
    print(f"Stock data directory exists: {os.path.exists(stock_data_dir)}")
    
    if os.path.exists(stock_data_dir):
        stock_files = [f for f in os.listdir(stock_data_dir) if f.endswith('.csv')]
        print(f"Stock files: {stock_files}")
    
    # Check twitter_data_new
    twitter_data_dir = os.path.join(data_dir, "twitter_data_new")
    print(f"\nTwitter data directory: {twitter_data_dir}")
    print(f"Twitter data directory exists: {os.path.exists(twitter_data_dir)}")
    
    if os.path.exists(twitter_data_dir):
        twitter_files = [f for f in os.listdir(twitter_data_dir) if f.endswith('.csv')]
        print(f"Twitter files: {twitter_files}")
else:
    print("Data directory not found!")
