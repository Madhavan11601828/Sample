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
    
    # Ensure classifier input matches combined_embeddings' shape
    x = self.classifier(combined_embeddings)
    return x
