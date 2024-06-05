import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(domain, model_name):
    try:
        model_path = os.path.join('model', domain)
        os.makedirs(model_path, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,force_download=True)
        print("Saving model and tokenizer to local directory...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print(f"Model and tokenizer loaded from {model_path}")
        return True

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return False
load_model(domain="bank", model_name="Khoa/bert-for-ads-cls-1")