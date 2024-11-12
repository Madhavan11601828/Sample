import hashlib
import logging
import os
from dataclasses import dataclass

import openai
import requests
import torch
from openai import AzureOpenAI
from safetensors.torch import load_file  # Import safetensors loader
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Set up logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI API credentials
AZURE_OPENAI_API_BASE = "https://genai-global-openai-dev.openai.azure.com/"
AZURE_OPENAI_API_KEY = "7dd2ab2bd3f3416ebbc41d53a8f1572c"
openai_api_key = AZURE_OPENAI_API_KEY
openai_api_base = AZURE_OPENAI_API_BASE
openai_api_version = "2024-02-15-preview"
openai_embed_api_version = "2023-05-15"

# Dictionary mapping model names to their IDs
MODEL_IDS = {
    "gpt-4o-mini": 0,
    "gpt-4o": 1,
}

# Configuration for matrix factorization (MF) model
GPT_4_AUGMENTED_CONFIG = {
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented/model.safetensors"},
}

# In-memory cache for storing embeddings to avoid redundant API calls
embedding_cache = {}

# Retry mechanism to handle transient errors, rate limits, and server errors with exponential backoff
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.RequestException)),
)
def call_openai_for_embedding(prompt):
    """Call Azure OpenAI API to get embedding for a prompt."""
    try:
        logger.info(f"Calling OpenAI API for embedding of prompt: {prompt}")
        response = requests.post(
            f"{openai_api_base}/openai/deployments/text-embedding-ada-002/embeddings?api-version={openai_embed_api_version}",
            headers={"api-key": openai_api_key},
            json={"input": [prompt], "model": "text-embedding-ada-002"},
        )
        response.raise_for_status()
        response_data = response.json()

        if response_data and "data" in response_data and len(response_data["data"]) > 0:
            return response_data
        else:
            raise ValueError("Unexpected response structure from OpenAI API")
    except Exception as e:
        logger.error(f"Failed to call OpenAI API: {e}")
        raise


# Cached version of prompt embedding retrieval with error handling
def get_prompt_embedding(prompt):
    """Get prompt embedding from Azure OpenAI API with caching."""
    if prompt in embedding_cache:
        logger.info(f"Using cached embedding for prompt: {prompt}")
        return embedding_cache[prompt]
    try:
        response = call_openai_for_embedding(prompt)
        if "data" in response and len(response["data"]) > 0 and "embedding" in response["data"][0]:
            embedding = torch.tensor(response["data"][0]["embedding"])
            embedding_cache[prompt] = embedding
            return embedding
        else:
            raise ValueError("Unexpected response structure from OpenAI API")
    except Exception as e:
        logger.error(f"Error while fetching embedding from OpenAI: {e}")
        raise


# Dataclass to store strong and weak models
@dataclass
class ModelPair:
    strong: str
    weak: str


# Matrix Factorization (MF) model class definition using PyTorch
class MFModel(torch.nn.Module):
    def __init__(self, dim, num_models, text_dim, num_classes, use_proj):
        super(MFModel, self).__init__()
        self.use_proj = use_proj
        self.proj = torch.nn.Linear(text_dim, dim) if use_proj else torch.nn.Identity()
        self.P = torch.nn.Embedding(num_models, dim)
        self.embedding_model = "text-embedding-ada-002"
        if self.use_proj:
            self.text_proj = torch.nn.Sequential(torch.nn.Linear(text_dim, dim, bias=False))
        else:
            if text_dim != dim:
                raise ValueError(f"text_dim {text_dim} must equal dim {dim} if not using projection")

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim + dim, num_classes),
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

        logger.info(f"Model embeddings shape: {model_embeddings.shape}")
        logger.info(f"Prompt embedding shape: {prompt_embed.shape}")

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


# Controller class that manages routing logic and uses MFModel
class Controller:
    def __init__(
        self, strong_model, weak_model, dim, num_models, text_dim, num_classes, use_proj, config=None
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        if config is None:
            config = GPT_4_AUGMENTED_CONFIG
        self.model = MFModel(dim=dim, num_models=64, text_dim=text_dim, num_classes=num_classes, use_proj=use_proj)
        self.model.load(config["mf"]["checkpoint_path"])
        self.model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def route(self, prompt_embedding, threshold):
        winrate = self.model.predict_win_rate(
            MODEL_IDS[self.model_pair.strong], MODEL_IDS[self.model_pair.weak], prompt_embedding
        )
        return "gpt-4o" if winrate >= threshold else "gpt-4o-mini"


# Instantiate the controller
controller_instance = Controller(
    strong_model="gpt-4o",
    weak_model="gpt-4o-mini",
    dim=128,
    num_models=64,
    text_dim=1536,
    num_classes=1,
    use_proj=True,
)


# Function to extract prompt and its embedding
def extract_prompt_and_embedding(request_data):
    try:
        logger.info(f"Request data received: {request_data}")
        prompt = request_data.get("prompt", "")
        messages = request_data.get("messages", [])

        if not prompt and messages:
            for message in messages:
                prompt += message.get("content", "")
            logger.info(f"Extracted prompt: {prompt}")

        if not prompt:
            raise ValueError("Prompt not found in the request data")

        prompt_embedding = get_prompt_embedding(prompt)
        logger.info("Successfully generated embedding for the prompt")
        return prompt_embedding, prompt, request_data
    except Exception as e:
        logger.error(f"Error during extraction of prompt and embedding: {e}")
        raise


# Function to handle chat completion
def chat_completion(request_data):
    try:
        prompt_embedding, prompt, request_data = extract_prompt_and_embedding(request_data)
        threshold = 0.1
        selected_model = controller_instance.route(prompt_embedding, threshold)

        logger.info(f"Using model: {selected_model} for prompt: {prompt}")
        client = AzureOpenAI(azure_endpoint=openai_api_base, api_key=openai_api_key, api_version=openai_api_version)
        response = client.chat.completions.create(
            model=selected_model,
            messages=request_data["messages"],
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 0.95),
            max_tokens=request_data.get("max_tokens", 800),
        )
        response_data = response.model_dump()
        response_data["selected_model"] = selected_model
        return response_data
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        return {"error": str(e)}


# Example request data
request_data = {
    "prompt": "Explain matrix factorization in machine learning.",
    "messages": [{"content": "What is matrix factorization?"}],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
}

# Run the chat completion in Databricks
response = chat_completion(request_data)
print(response)
