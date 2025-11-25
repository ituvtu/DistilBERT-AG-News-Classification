import json
import gradio as gr
from transformers import pipeline

MODEL_PATH = "ituvtu/distilbert-ag-news-classifier"
CATEGORY_MAPPING = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
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
        model_output = news_classifier(text, return_all_scores=True)
        predictions = model_output[0]

        results = {}
        for item in predictions:
            label_id = int(item["label"].split("_")[-1])
            category_name = CATEGORY_MAPPING.get(label_id, "Unknown")
            results[category_name] = item["score"]

        return results

    except Exception as error:
        return {str(error): 0.0}

example_data = load_example_headlines(EXAMPLES_FILE)

interface = gr.Interface(
    fn=classify_headline,
    inputs=gr.Textbox(lines=3, placeholder="Enter news headline here..."),
    outputs=gr.Label(),
    title="DistilBERT News Classifier",
    description="This model classifies news headlines into four categories: World, Sports, Business, and Sci/Tech.",
    examples=example_data
)

if __name__ == "__main__":
    interface.launch()