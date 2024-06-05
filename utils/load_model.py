import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(domain, model_name):
    try:
        model_path = os.path.join('model', domain)
        os.makedirs(model_path, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("Saving model and tokenizer to local directory...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print(f"Model and tokenizer loaded from {model_path}")
        return True

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return False

def load_model_from_disk(domain):
    try:
        model_path = os.path.join('model', domain)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return model, tokenizer
    except Exception as e:
        return None
    
