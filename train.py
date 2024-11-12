import json

# Function to load data from JSONL files
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

# Load all provided files
file_paths = ["/mnt/data/unify.jsonl", "/mnt/data/martian.jsonl", "/mnt/data/gpt-4_single.jsonl"]
data = load_data(file_paths)
print(f"Loaded {len(data)} data points.")



import torch
import torch.optim as optim
from safetensors.torch import save_file
from main import MFModel  # Ensure this points to where MFModel is defined

# Initialize the model
model = MFModel(dim=128, num_models=64, text_dim=1536, num_classes=1, use_proj=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define an optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification

# Assume that each data item has 'model_id', 'prompt_embedding', and 'label' fields
def train(model, data, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for item in data:
            # Prepare inputs and target
            model_ids = item['model_id']  # Example format, adjust based on actual data structure
            prompt_embed = torch.tensor(item['prompt_embedding']).float().to(device)
            label = torch.tensor([item['label']]).float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(model_ids, prompt_embed)
            loss = criterion(output, label)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log the loss for the current epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")

# Train the model
train(model, data, epochs=10)


# Save model weights
state_dict = model.state_dict()
checkpoint_path = "model.safetensors"
save_file(state_dict, checkpoint_path)
print(f"Model weights saved to {checkpoint_path}")
