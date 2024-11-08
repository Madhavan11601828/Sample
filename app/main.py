import hashlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from safetensors.torch import load_file  # Import safetensors loader

import logging
import os

from dataclasses import dataclass

from openai import AzureOpenAI

import openai

import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from safetensors.torch import load_file  # Import safetensors loader
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

load_dotenv()
# Set up logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Azure OpenAI API credentials from environment variables
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_version = "2024-02-15-preview"
openai_embed_api_version = "2023-05-15"

# Initialize FastAPI application
app = FastAPI()

# Define the filter
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.args and len(record.args) > 2 and (record.args[2] != '/health' or record.args[2] != '/')

# Add filter to the logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Dictionary mapping model names to their IDs
MODEL_IDS = {
    "gpt-4o-mini": 8,
    "gpt-4o": 1,
}

# Configuration for matrix factorization (MF) model
GPT_AUGMENTED_CONFIG = {
    "mf": {"checkpoint_path": "/app/routcllm/mf_gpt4_augmented/model.safetensors"},
}

# In memory cache for storing embeddings to avoid redundant API calls
embedding_cache = {}

# Retry mechanism to handle transient errors, rate limits, and server errors with exponential backoff
@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff for retries
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
        logger.error(f"Error while obtaining embedding for prompt: {e}")
        raise
