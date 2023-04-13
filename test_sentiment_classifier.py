from transformers import pipeline
import pytest

@pytest.fixture(scope="session")
def sentiment_pipeline():
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline


@pytest.mark.parametrize("example", ["I love you", "I love this",
                                      "I like you", "I feel good",
                                      "You did the right thing", "You are amazing!", "Thank you for your help."])
def test_obvious_positive_examples(sentiment_pipeline, example):
    results = sentiment_pipeline(example)
    for result in results:
        assert result["label"] == "POSITIVE", f"Expected POSITIVE, got {result['label']}"
        assert result["score"] >= 0.8 and result["score"] <= 1.0, f"Expected 1.0 >= score >= 0.8, got {result['score']}"



@pytest.mark.parametrize("example", ["I'm feeling really upset.", "This is a terrible situation.", 
                                      "I can't stand the way you're acting.", "That was a huge disappointment.", 
                                      "I'm really frustrated with the way things are going.", 
                                      "You're making me angry.", "I can't believe you would do something like that."])
def test_obvious_negative_examples(sentiment_pipeline, example):
    results = sentiment_pipeline(example)
    for result in results:
        assert result["label"] == "NEGATIVE", f"Expected NEGATIVE, got {result['label']}"
        assert result["score"] >= 0.8  and result["score"] <= 1.0, f"Expected 1.0>= score >= 0.8, got {result['score']}"


@pytest.mark.parametrize("example", [
    'I have been to the city of Paris and I did not like it there at all.',
    'I have been to the city of Tokyo and I did not like it there at all.',
    'I have been to the city of Sao Paulo and I did not like it there at all.',
    'I have been to the city of Moscow and I did not like it there at all.',
    'I have been to the city of Mumbai and I did not like it there at all.'
])
def test_city_invariance(sentiment_pipeline, example):
    results = sentiment_pipeline(example)
    for result in results:
        assert result["label"] == "NEGATIVE", f"Expected NEGATIVE, got {result['label']}"
        assert result["score"] >= 0.8  and result["score"] <= 1.0, f"Expected 1.0 >= score >= 0.8, got {result['score']}"



## THESE TESTS WILL FAIL.
@pytest.mark.parametrize("example", [
    "A group of flamingos is called a flamboyance.",
    "The shortest war in history was between Britain and Zanzibar in 1896.",
    "The world's oldest piece of chewing gum is over 9,000 years old.",
    "The Great Wall of China is not visible from space with the naked eye.",
    "The first webcam was created at the University of Cambridge to monitor the coffee pot in the computer science department.",
    "The shortest commercial flight in the world lasts only 1.5 minutes and travels from the Orkney Islands to Westray in Scotland.",
    "The first person to survive going over Niagara Falls in a barrel was a 63-year-old schoolteacher named Annie Edson Taylor.",
    "The longest wedding veil was longer than 63 football fields and was worn by a woman in Cyprus in 2018.",
    "The world's largest snowflake on record measured 15 inches wide and 8 inches thick."
])
def test_neutral_sentiment(sentiment_pipeline, example):
    results = sentiment_pipeline(example)
    for result in results:
        assert result["score"] < 0.65, f"Expected score <0.65 got {result['score']} with label {result['label']}"


