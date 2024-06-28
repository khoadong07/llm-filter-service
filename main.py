from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Trainer, TrainingArguments
import torch
from loguru import logger
import os
from dotenv import load_dotenv
from time import time
from helpers.response_template import success, bad_request
from model import PredictResponse, PredictRequest, SetupModelRequest
from utils.load_dataset import PredictDataset
from utils.load_model import load_model_from_disk, load_model
import json
from langdetect import detect

load_dotenv()

LOG_PATH = os.getenv("LOG_PATH")

logger.add(LOG_PATH + "{time}.log", rotation="1 day")

app = FastAPI()

# Set up CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
    # Add other allowed origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ModelManager class to manage loaded models
class ModelManager:
    def __init__(self):
        self.loaded_models = {}

    def load_model(self, model_name):
        if model_name not in self.loaded_models:
            try:
                model, tokenizer = load_model_from_disk(model_name)
                self.loaded_models[model_name] = (model, tokenizer)
                logger.info(f"Loaded model '{model_name}' successfully.")
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {str(e)}")

    def get_model(self, model_name):
        return self.loaded_models.get(model_name, (None, None))


model_manager = ModelManager()


# Function to perform prediction with labels
def predict_with_labels(model, tokenizer, input_data, labels, lang_detect=False, max_length=100, batch_size=32):
    contents_list = [item.content for item in input_data]
    ids_list = [item.id for item in input_data]
    start = time()

    # Tokenize new data for prediction
    new_encodings = tokenizer(contents_list, truncation=True, padding=True, max_length=max_length)
    new_dataset = PredictDataset(new_encodings)

    # Define a Trainer for prediction with appropriate batch size
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False
    )
    predict_trainer = Trainer(model=model, args=training_args)

    # Make predictions
    predictions = predict_trainer.predict(new_dataset)
    logits = predictions.predictions

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    end = time()
    processing_time = end - start
    results = []

    # Prepare results
    for i, prob in enumerate(probs):
        predict_result = []
        for label, prob_value in zip(labels, prob):
            predict_result.append({"label": label, "prob": float(prob_value)})
        results.append({
            "id": ids_list[i],
            "lang": detect(str(contents_list[i])) if lang_detect else None,
            "predict": predict_result
        })

    return results, processing_time


# Endpoint to setup model (currently placeholder)
@app.post("/api/filter-service/setup-model")
async def setup_model(model: SetupModelRequest):
    # Placeholder for actual setup logic
    try:
        load_model(domain=model.domain, model_name=model.model_name)
        return success("Model setup successfully", "")
    except Exception as e:
        logger.error(f"Failed to setup model '{model.model_name}': {str(e)}")
        return bad_request("Failed to setup model")


# Endpoint to perform prediction
@app.post("/api/filter-service/run")
async def predict(request: PredictRequest):
    logger.info(f"Starting filter service for domain: {request.domain}")
    start_time = time()
    input_data = request.contents
    domain = request.domain
    lang_detect = request.lang_detect
    labels = ["spam", "not_spam"]

    model_manager.load_model(domain)
    model, tokenizer = model_manager.get_model(domain)

    if model is None or tokenizer is None:
        return bad_request(f"Model '{domain}' not found or failed to load.", "")

    predictions, processing_time = predict_with_labels(model, tokenizer, input_data, labels, lang_detect)

    if predictions is None:
        return bad_request("Inference failed")

    end_time = time()
    request_time = end_time - start_time

    response = {
        "processing_time": processing_time,
        "request_time": request_time,
        "results": predictions
    }
    return success(message="Prediction successful", data=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
