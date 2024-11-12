class MFModel(torch.nn.Module):
    def __init__(self, dim=128, num_models=64, text_dim=1536, num_classes=1, use_proj=True):
        super(MFModel, self).__init__()
        self.use_proj = use_proj
        self.proj = torch.nn.Linear(text_dim, dim) if use_proj else torch.nn.Identity()
        
        # Define the embedding layer for model embeddings
        self.P = torch.nn.Embedding(num_models, dim)
        
        if self.use_proj:
            # Projection layer for prompt embeddings
            self.text_proj = torch.nn.Sequential(torch.nn.Linear(text_dim, dim, bias=False))

        # Retaining the original classifier structure
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_classes, num_classes),
        )

        # Additional layer to project concatenated embeddings back to dim for compatibility
        self.concat_proj = torch.nn.Linear(dim * 2, dim)

    def forward(self, model_ids, prompt_embed):
        # Apply projection to prompt_embed if enabled
        if self.use_proj:
            prompt_embed = self.proj(prompt_embed)

        if prompt_embed.dim() == 1:
            prompt_embed = prompt_embed.unsqueeze(0)
        
        model_embeddings = self.P(torch.tensor(model_ids))
        prompt_embed = prompt_embed.expand(model_embeddings.size(0), -1)
        
        # Concatenate embeddings along the feature dimension
        combined_embeddings = torch.cat((model_embeddings, prompt_embed), dim=1)
        
        # Project concatenated embeddings back to `dim` using concat_proj
        combined_embeddings = self.concat_proj(combined_embeddings)
        
        # Pass projected embeddings through the classifier
        x = self.classifier(combined_embeddings)
        return x
