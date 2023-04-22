import pytest
import os
from utils import is_garbage

@pytest.fixture(scope="session")
def tweets():
    """read tweets.txt file"""
    with open('tweets.txt', 'r') as f:
        tweets = f.readlines()
    return tweets


def test_is_garbage(tweets, threshold=0.2):
    """Check if the tweet is garbage."""
    garbage_tweets = [1 for tweet in tweets if is_garbage(tweet)]
    assert sum(garbage_tweets) / len(tweets) < threshold, f"Expected {threshold} >= garbage tweets ratio, got {len(garbage_tweets) / len(tweets)}"

