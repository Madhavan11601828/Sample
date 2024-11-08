import torch
import numpy as np
from main import MFModel, Controller, get_prompt_embedding, MODEL_IDS

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize your model configuration
dim = 128
num_models = 64
text_dim = 1024
num_classes = 2
use_proj = True
checkpoint_path = "/app/routellm/mf_gpt4_augmented/model.safetensors"

# Initialize the model and load checkpoint
model = MFModel(dim=dim, num_models=num_models, text_dim=text_dim, num_classes=num_classes, use_proj=use_proj)
model.load(checkpoint_path)
model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Calibration function for threshold
def calibrate_threshold(prompts, model_a_id, model_b_id, start=0.1, end=0.9, step=0.05):
    thresholds = np.arange(start, end, step)
    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        correct = 0
        total = len(prompts)

        for prompt in prompts:
            prompt_embedding = get_prompt_embedding(prompt)
            winrate = model.predict_win_rate(model_a_id, model_b_id, prompt_embedding)

            # Choose the model based on the current threshold
            selected_model = "gpt-4" if winrate >= threshold else "gpt-4o-mini"
            # Here, we assume "gpt-4" should win more often if the prompt is complex, based on some ground truth
            # For demonstration purposes, you may need to adjust the condition based on actual requirements

            if (winrate >= threshold and selected_model == "gpt-4") or (winrate < threshold and selected_model == "gpt-4o-mini"):
                correct += 1

        accuracy = correct / total
        logger.info(f"Threshold: {threshold}, Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    logger.info(f"Best threshold: {best_threshold} with accuracy: {best_accuracy}")
    return best_threshold

# Example usage with sample prompts
if __name__ == "__main__":
    # Define some example prompts to calibrate the threshold
    prompts = [
        "What is the weather like today?",
        "Explain the theory of relativity.",
        "What are the health benefits of drinking water?",
        "Describe the process of photosynthesis.",
        "What is quantum mechanics?"
    ]

    # Calibrate the threshold
    best_threshold = calibrate_threshold(prompts, MODEL_IDS["gpt-4o"], MODEL_IDS["gpt-4o-mini"])
    print(f"Optimal threshold determined: {best_threshold}")



import argparse
import yaml
import json
import pandas as pd
from datasets import load_dataset, Dataset

# Configuration for matrix factorization (MF) model
GET_AUGMENTED_CONFIG = {
    "checkpoint_path": "/app/routellm/mf_gpt4_augmented/model.safetensors",
}

# Function to calibrate the threshold for the matrix factorization model
def calibrate_threshold(routers, strong_model_pct):
    # Load the thresholds dataset from Hugging Face Hub
    thresholds_df = load_dataset(
        "routellm/lmsys-arena-human-preference-55k-thresholds", split="train"
    ).to_pandas()

    # Calibrate threshold for each router
    for router in routers:
        threshold = thresholds_df[router].quantile(q=1 - strong_model_pct)
        print(
            f"For {strong_model_pct * 100}% strong model calls for {router}, threshold = {round(threshold, 5)}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strong-model-pct", type=float, required=True,
        help="Percentage of strong model calls to calibrate the threshold for."
    )
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=["matrix_factorization"],
        help="List of routers to calibrate thresholds for."
    )
    args = parser.parse_args()

    calibrate_threshold(args.routers, args.strong_model_pct)
