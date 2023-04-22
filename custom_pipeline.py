import transformers
import datasets

class SentimentClassifierPipeline(transformers.TextClassificationPipeline):
    """
    Sentiment classification pipeline that accepts a model and a tokenizer
    """

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.compute_metrics = datasets.load_metric("accuracy")

    def __call__(self, inputs, **kwargs):
        """
        Accepts a string and returns a dictionary with the following keys:
        - label: the sentiment label (either "POSITIVE" or "NEGATIVE")
        - score: the probability associated with the label
        """
        # Run the model
        output = self.model(**self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, **kwargs))

        # Get the label with the highest score
        label = self.model.config.id2label[output.logits.argmax().item()]

        # Get the score for this label
        score = output.logits.softmax(dim=1)[0, output.logits.argmax().item()].item()

        # Return the label and the score
        return {"label": label, "score": score}

    def load_model(self, model_path):
        """
        Loads the model from the given path
        """
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def randomly_initialize_model(self):
        """
        Randomly initialize the model
        """
        self.model = transformers.AutoModelForSequenceClassification.from_config(self.model.config)
    
    def evaluate(self, eval_dataset, **kwargs):
        """
        Evaluate the model on the given evaluation dataset and return accuracy
        """
        self.model = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(**kwargs),
            tokenizer=self.tokenizer,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        ).evaluate()
        return self.model.metrics["eval_accuracy"]
        


if __name__ == "__main__":

    # Load the tokenizer and the model
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Create the pipeline
    pipeline = SentimentClassifierPipeline(model=model, tokenizer=tokenizer)

    # Test the pipeline
    print(pipeline("I love you"))
    print(pipeline("I hate you"))