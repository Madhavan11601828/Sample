import requests
import logging

class AppClient:
    def __init__(self, base_url, api_key=None):
        """
        Initialize the wrapper for the application.

        :param base_url: Base URL of the deployed application.
        :param api_key: Optional API key for authentication (if required).
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.logger = logging.getLogger(__name__)

    def chat_completion(self, x_use_case, prompt, messages=None, temperature=0.7, top_p=0.95, max_tokens=800):
        """
        Call the chat completion API with the X-USE_CASE header.

        :param x_use_case: The use case to pass in the X-USE_CASE header.
        :param prompt: Prompt for the chat completion.
        :param messages: Optional list of messages for conversation history.
        :param temperature: Sampling temperature for the model.
        :param top_p: Top-p sampling value.
        :param max_tokens: Maximum number of tokens to generate.
        :return: Response from the API.
        """
        endpoint = f"{self.base_url}/llm/openai/deployments/smart-router/chat/completions"
        headers = self.headers.copy()
        headers["X-USE_CASE"] = x_use_case

        payload = {
            "prompt": prompt,
            "messages": messages or [],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            self.logger.info("Sending request to the chat completion endpoint.")
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for HTTP error codes
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling chat completion API: {e}")
            raise

    def health_check(self):
        """
        Check the health of the deployed application.

        :return: Status of the application.
        """
        endpoint = f"{self.base_url}/health"
        try:
            self.logger.info("Checking health of the application.")
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error checking health: {e}")
            raise
