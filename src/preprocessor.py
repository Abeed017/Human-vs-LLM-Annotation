import re
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_tweet(text, language="id"):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    return text


def build_dataset(df, task="sentiment"):
    if "text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column")

    df = df.copy()
    df["text"] = df["text"].apply(preprocess_tweet)
    df = df[df["text"].str.len() > 5]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)
    return df


def train_val_test_split(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")
    
    n_samples = len(df)
    
    # For small datasets, use a different strategy
    if n_samples < 10:
        # Ensure at least 1 sample in each split, with priority to train set
        min_train = max(1, int(n_samples * train_size))
        min_val = max(1, int(n_samples * val_size))
        min_test = n_samples - min_train - min_val
        
        # Adjust if test set is 0
        if min_test <= 0:
            min_test = 1
            min_val = max(1, n_samples - min_train - 1)
        
        # First split: separate train from (val + test)
        temp_size = min_val + min_test
        train, temp = train_test_split(df, train_size=min_train, random_state=random_state)
        
        # Second split: separate val from test
        if len(temp) > 1:
            val, test = train_test_split(temp, train_size=min_val, random_state=random_state)
        else:
            val = temp
            test = temp.iloc[0:0]  # Empty dataframe with same structure
    else:
        train, temp = train_test_split(df, test_size=(1 - train_size), random_state=random_state)
        val, test = train_test_split(temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
