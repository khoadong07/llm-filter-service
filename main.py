from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Trainer, TrainingArguments
import torch
from loguru import logger
import os 
from dotenv import load_dotenv
from time import time

from model import PredictResponse, PredictRequest, SetupModelRequest
from utils.load_dataset import PredictDataset
from utils.load_model import load_model

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

def predict_with_labels(model, tokenizer, input_data, labels, max_length=100, batch_size=16):
    start = time()

    # Tokenize new data for prediction
    new_encodings = tokenizer(input_data, truncation=True, padding=True, max_length=max_length)
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
        for label, prob_value in zip(labels, prob):
            results.append({"id": i, "label": label, "prob": prob_value})

    return results, processing_time


@app.post("/api/filter-service/setup-model", response_model=SetupModelRequest)
async def setup_model(model: SetupModelRequest):
    try:
        res = load_model(domain=model.domain, model_name=model.model_name)
    except:


@app.post("/api/filter-service/run", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time()
    input_data = request.content
    labels = ["ads", "none_ads"]
    predictions, processing_time = predict_with_labels(model, tokenizer, input_data, labels)
    end_time = time()
    request_time = end_time - start_time

    response = {
        "id": request.id,
        "processing_time": processing_time,
        "request_time": request_time,
        "results": predictions
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)