import pytest
from utils import is_garbage, remove_numbers, remove_punctuation


@pytest.mark.parametrize("tweet, cleaned_tweet", [("Hello, world!", "Hello world"), ("!@$%^", "")])
def test_remove_punctuation(tweet, cleaned_tweet):
    """Test if punctuation is removed."""
    assert remove_punctuation(tweet) == cleaned_tweet, f"Expected {cleaned_tweet} but got {remove_punctuation(tweet)}"



@pytest.mark.parametrize("tweet, cleaned_tweet", [("Hello123!", "Hello!"), ("12345", ""), ("Hello 1239", "Hello ")])
def test_remove_numbers(tweet, cleaned_tweet):
    """Test if punctuation is removed."""
    assert remove_numbers(tweet) == cleaned_tweet, f"Expected {cleaned_tweet} but got {remove_numbers(tweet)}"
