import gradio as gr
from transformers import pipeline
import json

model_name = "ituvtu/distilbert-ag-news-classifier" 

id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

try:
    pipe = pipeline("text-classification", model=model_name)
    print(f"Model {model_name} has been successfully downloaded.")
except Exception as e:
    model_load_error = str(e)
    print(f"Model loading error: {model_load_error}")


def predict_news_category(text):
    
    if pipe is None:
        return {f"Critical error: Model failed to load.": 0.0,
                f"Error: {model_load_error}": 0.0}

    if not text or text.strip() == "":
        return {"Enter text for analysis": 0.0}

    try:
        predictions = pipe(text, return_all_scores=True)
    except Exception as e:
        return {f"Error during forecasting: {e}": 0.0}

    formatted_output = {}
    try:
        if predictions and predictions[0]:
            for pred in predictions[0]:
                readable_label = pred['label'] 
                score = pred['score']
                formatted_output[readable_label] = score
        else:
            return {"The model did not return a result.": 0.0}
            
    except Exception as e:
        return {f"Output formatting error: {e}": 0.0}

    return formatted_output

def load_examples(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        print(f"Examples from {file_path} have been successfully downloaded.")
        return examples
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found.")
        return [["Error: example file not found."]]
    except Exception as e:
        print(f"ERROR when reading {file_path}: {e}")
        return [[f"JSON reading error: {e}"]]
examples = load_examples("examples.json")

# Interface
title = "News classifier (AG News)"
description = "Enter a news headline, and the DistilBERT model (trained on AG News) will determine its category. Try one of the 12 examples below to test the model!"

iface = gr.Interface(
    fn=predict_news_category,
    inputs=gr.Textbox(lines=3, placeholder="Enter the news headline here..."),
    outputs=gr.Label(), 
    title=title,
    description=description,
    examples=examples
)

# Launch 
iface.launch()