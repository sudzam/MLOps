# Summary: Handle data loading, preprocessing, and simulating drift for sentiment analysis using GPT-2

import pandas as pd
from datasets import load_dataset # type: ignore
from transformers import GPT2Tokenizer
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from config import DATA_DIR, MODEL_NAME, MAX_LENGTH

# Load and prepare sentiment analysis dataset
def load_and_prepare_data():
    # Load IMDB dataset
    print("Loading dataset IMDB dataset")
    dataset = load_dataset("imdb", split="train[:1000]")
    
    # Convert to pandas for future processing
    # existing Column: text, label. New column: sentiment
    df = pd.DataFrame(dataset)
    df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})
    
    # Save raw data
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "raw_data.csv"), index=False)
    
    return df

# Preprocess text data for GPT-2 fine-tuning
def preprocess_data(df):  
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create formatted text for fine-tuning
    # GPT-2 expects <|endoftext|> to demarcate examples
    def format_example(row):
        return f"Review: {row['text']}\nSentiment: {row['sentiment']}<|endoftext|>"
    
    df['formatted_text'] = df.apply(format_example, axis=1)
    
    # Save processed data
    processed_path = os.path.join(DATA_DIR, "processed_data.csv")
    df.to_csv(processed_path, index=False)
    
    return df, processed_path

# Simulate data drift by modifying
def simulate_drift_data(original_df, drift_factor=0.3):
    drift_df = original_df.copy()
    
    # Simulate drift by shortening reviews (concept drift)
    mask = drift_df.sample(frac=drift_factor).index
    drift_df.loc[mask, 'text'] = drift_df.loc[mask, 'text'].str[:100]
    
    # Save drift data
    drift_path = os.path.join(DATA_DIR, "drift_data.csv")
    drift_df.to_csv(drift_path, index=False)
    
    return drift_df, drift_path