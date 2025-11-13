import gradio as gr
from transformers import pipeline


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
                label_num_str = pred['label']
                score = pred['score']
                label_index = int(label_num_str.split('_')[-1])
                readable_label = id2label.get(label_index, "Unknown")
                formatted_output[readable_label] = score
        else:
            return {"The model did not return a result.": 0.0}
            
    except Exception as e:
        return {f"Output formatting error: {e}": 0.0}

    return formatted_output


examples = [
    # 0: World
    ["Global leaders meet for climate change summit."],
    ["Tensions rise in Middle East after new treaty."],
    ["Brexit trade deal negotiations stall over fishing rights."], 
    
    # 1: Sports
    ["Olympic athlete breaks world record in 100m sprint."],
    ["Local football team wins championship."],
    ["Manchester United announces record-breaking sponsorship deal."], 
    
    # 2: Business
    ["Federal Reserve to announce interest rate decision."],
    ["Stock market hits all-time high amid tech boom."],
    ["Tesla unveils new battery technology, stock soars."], 
    
    # 3: Sci/Tech
    ["New discovery on Mars could change everything."],
    ["Researchers develop new AI capable of writing code."],
    ["Massive cybersecurity breach exposes government data."]
]

# Interface
title = "News classifier (AG News)"
description = "Enter a news headline, and the DistilBERT model (trained on AG News) will determine its category. Try one of the 12 examples below to test the model!"

iface = gr.Interface(
    fn=predict_news_category,
    
    inputs=gr.Textbox(lines=3, placeholder="Enter the headline of the news here..."),
    
    outputs=gr.Label(), 
    
    title=title,
    description=description,
    examples=examples
)

# Launch 
iface.launch()