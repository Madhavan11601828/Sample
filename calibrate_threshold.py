import torch
import hashlib
from safetensors.torch import load_file
import logging

class MFModel(torch.nn.Module):
    def __init__(self, dim=128, num_models=64, text_dim=1536, num_classes=1, use_proj=True):
        super(MFModel, self).__init__()
        self.use_proj = use_proj
        self.proj = torch.nn.Linear(text_dim, dim) if use_proj else torch.nn.Identity()
        
        # Updated to match checkpoint dimensions
        self.P = torch.nn.Embedding(num_models, dim)
        
        if self.use_proj:
            # Updated to match checkpoint dimensions
            self.text_proj = torch.nn.Sequential(torch.nn.Linear(text_dim, dim, bias=False))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_classes, num_classes),
        )

    def forward(self, model_ids, prompt_embed):
        if self.use_proj:
            prompt_embed = self.proj(prompt_embed)

        if prompt_embed.dim() == 1:
            prompt_embed = prompt_embed.unsqueeze(0)
        
        model_embeddings = self.P(torch.tensor(model_ids))
        prompt_embed = prompt_embed.expand(model_embeddings.size(0), -1)
        
        combined_embeddings = torch.cat((model_embeddings, prompt_embed), dim=1)
        x = self.classifier(combined_embeddings)
        return x

    @torch.no_grad()
    def predict_win_rate(self, model_a, model_b, prompt_embed):
        logits = self.forward([model_a, model_b], prompt_embed)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path, expected_checksum=None):
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        file_checksum = hasher.hexdigest()
        if expected_checksum and file_checksum != expected_checksum:
            raise ValueError("Checksum verification failed for the model file")
        
        tensor_dict = load_file(path)
        tensor_dict = {k.replace("classifier.0.", "classifier."): v for k, v in tensor_dict.items()}
        self.load_state_dict(tensor_dict, strict=False)

    def calibrate_threshold(self, prompts, model_a_id, model_b_id, start=0.1, end=0.9, step=0.05):
        """Calibrate the threshold for choosing between two models based on prompt embeddings."""
        best_threshold = start
        highest_accuracy = 0

        for threshold in torch.arange(start, end, step):
            correct_predictions = 0

            for prompt in prompts:
                # Get embedding for the prompt
                prompt_embedding = get_prompt_embedding(prompt)

                # Predict win rate for the models using matrix factorization
                logits = self.forward(
                    model_ids=[model_a_id, model_b_id], prompt_embed=prompt_embedding
                )

                if logits.numel() == 2:
                    winrate = torch.sigmoid(logits[0] - logits[1]).item()
                elif logits.numel() == 1:
                    winrate = torch.sigmoid(logits).item()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                # Check if the prediction is correct based on the threshold
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
