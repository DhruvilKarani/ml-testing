import pytest
import os
from datasets import load_dataset
import transformers
from custom_pipeline import SentimentClassifierPipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@pytest.fixture(scope="session")
def incremental_sentiment_pipeline():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    incremental_sentiment_pipeline = SentimentClassifierPipeline(model, tokenizer)
    return incremental_sentiment_pipeline


@pytest.fixture(scope="session")
def random_sentiment_pipeline():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    random_sentiment_pipeline = SentimentClassifierPipeline(model, tokenizer)
    random_sentiment_pipeline.randomly_initialize_model()
    return random_sentiment_pipeline


@pytest.fixture(scope="session")
def eval_dataset():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    def preprocess_function(examples):
       return tokenizer(examples["text"], truncation=True)

    eval_dataset = load_dataset("imdb", split="test").shuffle(seed=42).select([i for i in list(range(3000))])
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    return eval_dataset


def test_incremental_pipeline_better_than_random(eval_dataset, incremental_sentiment_pipeline, random_sentiment_pipeline):
    incremental_accuracy = incremental_sentiment_pipeline.evaluate(eval_dataset, eval_steps=1, output_dir="./results") #TODO: change steps
    random_accuracy = random_sentiment_pipeline.evaluate(eval_dataset, eval_steps=1, output_dir="./results") #TODO: change steps
    # Check that the incremental pipeline is better than the random pipeline
    assert incremental_accuracy > random_accuracy, f"Expected incremental accuracy {incremental_accuracy} > random accuracy {random_accuracy}"

