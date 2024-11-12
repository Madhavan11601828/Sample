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
            # Adjust the field names based on actual data structure
            model_ids = [0] if item['model_id'] == 'mistral-8x7b-instruct-v0.1@fireworks-ai' else [1]  # Example encoding of model_id

            # Retrieve the text from 'choices[0]["turns"]' to use as the input prompt
            prompt_text = item['choices'][0]['turns'][0]
            prompt_embed = torch.rand(1536).float().to(device)  # Placeholder random embedding for the text

            # Use a dummy label
            label = torch.tensor([1.0]).float().to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(model_ids, prompt_embed)
            output = output.squeeze(dim=-1)  # Squeeze only the last dimension
            
            # Debugging statements to check shapes
            print(f"Output shape: {output.shape}, Label shape: {label.shape}")
            
            # Reshape label to match the output shape if needed
            if output.shape != label.shape:
                label = label.view_as(output)
                print(f"Reshaped label to: {label.shape}")

            # Calculate loss
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
