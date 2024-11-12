import argparse
import json
import logging
import torch
from main import get_prompt_embedding, MODEL_IDS, MFModel, GPT_4_AUGMENTED_CONFIG
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load the matrix factorization model with checkpoint settings
def load_matrix_factorization_model():
    model = MFModel(
        dim=128,              # Match checkpoint dimensions
        num_models=64,
        text_dim=1536,        # Match checkpoint dimensions
        num_classes=1,
        use_proj=True,
    )
    checkpoint_path = GPT_4_AUGMENTED_CONFIG["mf"]["checkpoint_path"]
    tensor_dict = load_file(checkpoint_path)
    
    try:
        # Attempt to load state dict and log mismatches
        missing_keys, unexpected_keys = model.load_state_dict(tensor_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
            
    except RuntimeError as e:
        logger.error(f"RuntimeError in loading state_dict: {e}")
        raise e

    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

# Function to calibrate threshold
def calibrate_threshold(prompts, model_a_id, model_b_id, start=0.1, end=0.9, step=0.05):
    model = load_matrix_factorization_model()

    best_threshold = start
    highest_accuracy = 0

    for threshold in torch.arange(start, end, step):
        correct_predictions = 0

        for prompt in prompts:
            # Get embedding for the prompt
            prompt_embedding = get_prompt_embedding(prompt)

            # Predict win rate for the models using matrix factorization
            with torch.no_grad():
                logits = model.forward(
                    model_ids=[model_a_id, model_b_id], prompt_embed=prompt_embedding
                )

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
