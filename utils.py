import string


def remove_punctuation(text):
    """Remove punctuation from a text."""
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def remove_numbers(text):
    """Remove numbers from a text."""
    for number in string.digits:
        text = text.replace(number, '')
    return text


def is_garbage(tweet):
    """Check if the tweet is garbage."""
    processed_tweet = remove_numbers(
        remove_punctuation(
            tweet.lower()
        )
    ).strip()
    if len(processed_tweet) < 5 and len(processed_tweet.split()) < 2:
        return True
