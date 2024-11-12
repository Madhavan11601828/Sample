import argparse
import json
import logging
import torch
from main import MFModel  # Import MFModel definition from your main file
from safetensors.torch import load_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load the model with weights from safetensors
def load_trained_model(safetensors_path):
    logger.info("Loading trained model.")
    model = MFModel(
        dim=128,              # Match dimensions from trained model
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    )
    # Load the state dict from safetensors
    tensor_dict = load_file(safetensors_path)
    model.load_state_dict(tensor_dict, strict=False)

    # Move to the appropriate device
    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

# Function to calibrate the threshold
def calibrate_threshold(prompts, model, model_a_id, model_b_id, start=0.1, end=0.9, step=0.05):
    logger.info("Starting threshold calibration.")
    best_threshold = start
    highest_accuracy = 0

    for threshold in torch.arange(start, end, step):
        correct_predictions = 0
        logger.info(f"Testing threshold: {threshold}")

        for prompt in prompts:
            # Generate a dummy prompt embedding (replace with real embeddings if available)
            prompt_embed = torch.rand(1536).float().to(model.device)  # Placeholder embedding

            # Predict win rate for model comparisons using matrix factorization
            with torch.no_grad():
                logits = model.forward(model_ids=[model_a_id, model_b_id], prompt_embed=prompt_embed)

                # Compute winrate from logits difference for model comparison
                if logits.numel() == 2:
                    winrate = torch.sigmoid(logits[0] - logits[1]).item()
                elif logits.numel() == 1:
                    winrate = torch.sigmoid(logits).item()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                logger.info(f"Winrate for current threshold {threshold}: {winrate}")

            # Determine if the prediction is correct based on the threshold
            if (winrate >= threshold and model_a_id == 0) or (winrate < threshold and model_b_id == 1):
                correct_predictions += 1

        # Calculate accuracy for the current threshold
        accuracy = correct_predictions / len(prompts)
        logger.info(f"Accuracy for threshold {threshold}: {accuracy}")

        # Update best threshold if the current accuracy is higher
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_threshold = threshold

    logger.info(f"Best threshold: {best_threshold}, Best accuracy: {highest_accuracy}")
    return best_threshold, highest_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True, help="Path to the JSON file with prompts")
    parser.add_argument("--start", type=float, default=0.1, help="Start value for threshold")
    parser.add_argument("--end", type=float, default=0.9, help="End value for threshold")
    parser.add_argument("--step", type=float, default=0.05, help="Step size for threshold")
    parser.add_argument("--model_path", type=str, default="trained_model.safetensors", help="Path to the trained model file")
    args = parser.parse_args()

    # Load prompts from JSON file
    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts for calibration.")

    # Load trained model
    model = load_trained_model(args.model_path)

    # Define model IDs for calibration
    model_a_id, model_b_id = 0, 1  # Example IDs, adjust based on actual use

    # Calibrate the threshold
    best_threshold, best_accuracy = calibrate_threshold(
        prompts, model, model_a_id, model_b_id, start=args.start, end=args.end, step=args.step
    )

    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")
