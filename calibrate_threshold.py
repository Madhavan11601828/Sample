import argparse
import torch
import yaml
import logging
import json
from main import get_prompt_embedding, MODEL_IDS, MFModel, GET_AUGMENTED_CONFIG
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the matrix factorization model
def load_matrix_factorization_model():
    model = MFModel(
        dim=128,  # Adjusted to match the checkpoint dimensions
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    )
    checkpoint_path = GET_AUGMENTED_CONFIG["checkpoint_path"]
    tensor_dict = load_file(checkpoint_path)
    model.load_state_dict(tensor_dict, strict=False)
    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

# Calibration function for threshold
def calibrate_threshold(prompts, model_a_id, model_b_id, start=0.1, end=0.9, step=0.05):
    model = load_matrix_factorization_model()

    best_threshold = start
    highest_accuracy = 0

    for threshold in range(int(start * 100), int(end * 100), int(step * 100)):
        threshold = threshold / 100.0
        correct_predictions = 0

        for prompt in prompts:
            # Get embedding for the prompt
            prompt_embedding = get_prompt_embedding(prompt)

            # Predict win rate for the models using matrix factorization
            with torch.no_grad():
                logits = model.forward(
                    model_ids=[model_a_id, model_b_id], prompt_embed=prompt_embedding
                )

                # Handle multi-element logits
                if logits.dim() > 1:
                    logits = logits.squeeze()

                # Calculate winrate using logits difference
                if logits.numel() == 2:
                    winrate = torch.sigmoid(logits[0] - logits[1]).item()
                elif logits.numel() == 1:
                    winrate = torch.sigmoid(logits).item()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

            # Determine if the prediction is correct based on the threshold
            if (winrate >= threshold and model_a_id == MODEL_IDS["gpt-4o"]) or \
               (winrate < threshold and model_b_id == MODEL_IDS["gpt-4o-mini"]):
                correct_predictions += 1

        # Calculate accuracy for the current threshold
        accuracy = correct_predictions / len(prompts)

        # Update best threshold if the current accuracy is higher
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, highest_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True, help="Path to the JSON file with prompts")
    parser.add_argument("--start", type=float, default=0.1, help="Start value for threshold")
    parser.add_argument("--end", type=float, default=0.9, help="End value for threshold")
    parser.add_argument("--step", type=float, default=0.05, help="Step size for threshold")
    args = parser.parse_args()

    # Load prompts from a JSON file
    with open(args.prompts, "r") as f:
        prompts = json.load(f)

    # Define model IDs for calibration
    model_a_id = MODEL_IDS["gpt-4o"]
    model_b_id = MODEL_IDS["gpt-4o-mini"]

    # Calibrate the threshold
    best_threshold, best_accuracy = calibrate_threshold(
        prompts, model_a_id, model_b_id, start=args.start, end=args.end, step=args.step
    )

    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")
