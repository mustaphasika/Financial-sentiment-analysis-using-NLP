from fastapi import FastAPI
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
import torch

# Specify the directory path where you saved the model in Google Drive
load_directory_drive = 'C:/Users/Midou/Desktop/Saclay/NLP/your_model_directory'


# Load the tokenizer and the model from Google Drive
loaded_tokenizer = AutoTokenizer.from_pretrained(load_directory_drive)
loaded_bert_model = AutoModelForSequenceClassification.from_pretrained(load_directory_drive)

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict")
def predict(text: str):
    # Specify the input sentence
    input_sentence = text

    # Tokenize and encode input text
    inputs = loaded_tokenizer(input_sentence, return_tensors="pt", truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = loaded_bert_model(**inputs)

        logits = outputs.logits

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Post-processing
        predicted_class_label = "Neutral" if predicted_class == 1 else ("Negative" if predicted_class == 0 else "Positive")

        # Print the result for the input sentence
        prediction_result = {"text": text, "prediction": predicted_class_label}

    return prediction_result
