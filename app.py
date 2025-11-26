import json
import gradio as gr
from transformers import pipeline

MODEL_PATH = "ituvtu/distilbert-ag-news-classifier"
EXAMPLES_FILE = "examples.json"

try:
    news_classifier = pipeline("text-classification", model=MODEL_PATH)
except Exception as error:
    news_classifier = None
    initialization_error = str(error)

def load_example_headlines(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []

def classify_headline(text):
    if news_classifier is None:
        return {"Model failed to initialize": 0.0}

    if not text or not text.strip():
        return {"Please enter valid text": 0.0}

    try:
        predictions = news_classifier(text, top_k=None)
        results = {}

        for item in predictions:
            category_name = item["label"]
            score = item["score"]
            results[category_name] = score

        return results

    except Exception as error:
        return {f"Error: {str(error)}": 0.0}

example_data = load_example_headlines(EXAMPLES_FILE)

interface = gr.Interface(
    fn=classify_headline,
    inputs=gr.Textbox(lines=3, placeholder="Enter news headline here..."),
    outputs=gr.Label(),
    title="DistilBERT News Classifier",
    description="This model classifies news headlines into four categories: World, Sports, Business, and Sci/Tech.",
    examples=example_data,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch()