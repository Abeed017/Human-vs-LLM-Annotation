import os
import pandas as pd

try:
    import tweepy
except ImportError:
    tweepy = None


def collect_tweets(bearer_token, language="id", count=500):
    """Collect recent tweets using Twitter/X API v2."""
    if tweepy is None:
        raise ImportError("tweepy is required for tweet collection. Install with pip install tweepy")

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    query = f"lang:{language} -is:retweet"
    tweets = []
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["text", "lang"],
        max_results=100,
    )

    for page in paginator:
        if page.data is None:
            continue
        for tweet in page.data:
            tweets.append({"text": tweet.text, "lang": tweet.lang})
            if len(tweets) >= count:
                break
        if len(tweets) >= count:
            break

    return pd.DataFrame(tweets)


def load_dataset(csv_path):
    """Load a CSV dataset with a 'text' column."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)
