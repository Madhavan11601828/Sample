import hashlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi_retry import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from safetensors.torch import load_file  # Import safetensors loader

import logging
import os

from dataclasses import dataclass

from openai import AzureOpenAI

import openai

import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from safetensors.torch import load_file  # Import safetensors loader
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

load_dotenv()

# Configuration for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Azure OpenAI API credentials from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
openai_embed_api_version = os.getenv("OPENAI_EMBED_API_VERSION")

# Initialize FastAPI application
app = FastAPI()

# Define the filter
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord)  -> bool:
        return record.args and len(record.args) >= 3 and (record.args[2]!="/health" or record.arg[2]!= "/"

# Add filter to the logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Dictionary mapping model names to their IDs
MODEL_IDS = {
    "gpt-4o-mini": 0,
    "gpt-4o": 1
}

# Configuration for matrix factorization (MF) model
GET_AUGMENTED_CONFIG = {
    "checkpoint_path": "/app/routellm/mf_gpt4_augmented/model.safetensors",
}

# In-memory cache for storing embeddings to avoid redundant API calls
embedding_cache = {}

# Retry mechanism to handle transient errors, rate limits, and server errors with exponential backoff
@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff for retry
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.RequestException)),
)
def call_openai_for_embedding(prompt):
    """ Call Azure OpenAI API to get embedding for a prompt """
    try:
        logger.info("Calling OpenAI API for embedding for prompt: {}".format(prompt))
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
        logger.error("Failed to call OpenAI API: {}".format(e))
        raise

# Cached version of prompt embedding retrieval with in-memory caching
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
@dataclass
class ModelPair:
    strong: str
    weak: str

# Matrix Factorization (MF) model class definition using PyTorch
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
    def __init__(self, router, strong_model, weak_model, dim, num_models, text_dim, num_classes, use_proj, config=None):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        if config is None:
            config = GPT4_AUGMENTED_CONFIG
        self.model = MFModel(dim=dim, num_models=64, text_dim=text_dim, num_classes=num_classes, use_proj=use_proj)
        self.model.load(config["mf"]["checkpoint_path"])
        self.model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def route(self, prompt_embedding, threshold):
        winner_model = self.strong_model if prompt_embedding > threshold else self.weak_model
        return winner_model

controller_instance = Controller(
    router="smart-router",
    strong_model="gpt-4",
    weak_model="gpt-4.0-weak",
    text_dim=1024,
    num_classes=2,
    use_proj=True,
)

# Helper function to extract prompt and its embedding from the request
async def extract_prompt_and_embedding(request):
    try:
        request_data = await request.json()
        logger.info(f"Request data received: {request_data}")
        prompt = request_data.get("prompt")
        if not prompt:
            prompt = ""
            messages = request_data.get("messages", [])
            for i in range(len(messages)):
                if "content" in messages[i]:
                    content_list = messages[i]["content"]
                    logger.info(f"Extracting prompt from messages: {content_list}")
                    content = " ".join([item.get("text", "") for item in content_list if "text" in item])
                    prompt += content
            logger.info(f"Extracted prompt: {prompt}")

        if not prompt:
            logger.error("Prompt not found in the request data")
            raise ValueError("Prompt not found in the request data")
        
        logger.info("Generating embedding for the prompt")
        prompt_embedding = get_prompt_embedding(prompt)
        logger.info("Successfully generated embedding for the prompt")
        return prompt_embedding, prompt, request_data
    except Exception as e:
        logger.exception(f"Error during extraction of prompt and embedding: {e}")
        raise

# Create an endpoint for chat completions using the smart-router
@app.post("/llm/openai/deployments/smart-router/chat/completions")
async def chat_completion(request: Request):
    """POST endpoint to handle chat completions.
    This endpoint routes the request through the controller's logic to select the appropriate model.
    """
    try:
        prompt_embedding, prompt, request_data = await extract_prompt_and_embedding(request)
        threshold = 0.5
        selected_model = controller_instance.route(prompt_embedding, threshold)

        logger.info(f"Using model: {selected_model} for prompt: {prompt}")
        client = AzureOpenAI(api_base=openai_api_base, api_key=openai_api_key, api_version=openai_api_version)
        response = client.chat_completion.create(
            model=selected_model,
            messages=request_data["messages"],
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 0.95),
            max_tokens=request_data.get("max_tokens", 800),
        )
        
        response_data = response.model_dump()
        response_data["selected_model"] = selected_model

        return JSONResponse(content=response_data, headers={"X-LLM-Model": selected_model})
    except Exception as e:
        logger.exception("Error during chat completion", exc_info=e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "online"})
