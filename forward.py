class MFModel(torch.nn.Module):
    def __init__(self, dim=128, num_models=64, text_dim=1536, num_classes=1, use_proj=True):
        super(MFModel, self).__init__()
        self.use_proj = use_proj
        self.proj = torch.nn.Linear(text_dim, dim) if use_proj else torch.nn.Identity()
        
        # Ensure P weight dimension matches checkpoint
        self.P = torch.nn.Embedding(num_models, dim)
        
        if self.use_proj:
            # Adjusted to match checkpoint dimensions
            self.text_proj = torch.nn.Sequential(torch.nn.Linear(text_dim, dim, bias=False))

        # Adjusted classifier dimensions to match combined_embeddings shape
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, num_classes),   # Using dim * 2 to match combined_embeddings shape
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
        
        # Adjusting concatenation to match expected classifier input dimensions
        combined_embeddings = torch.cat((model_embeddings, prompt_embed), dim=1)
        
        # Check the shape of combined_embeddings to confirm
        logger.info(f"Shape of combined_embeddings: {combined_embeddings.shape}")
        
        # Pass combined_embeddings through the classifier
        x = self.classifier(combined_embeddings)
        return x
