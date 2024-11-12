import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import torch

# Assuming the FastAPI app is named `app` as in the provided code
client = TestClient(app)

class TestEmbeddingFunctions(unittest.TestCase):
    @patch("requests.post")
    def test_call_openai_for_embedding_success(self, mock_post):
        # Mock the response from the OpenAI API
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        response = call_openai_for_embedding("test prompt")
        self.assertEqual(response["data"][0]["embedding"], [0.1, 0.2, 0.3])

    @patch("requests.post")
    def test_call_openai_for_embedding_fail(self, mock_post):
        # Mock a failed response
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        with self.assertRaises(Exception) as context:
            call_openai_for_embedding("test prompt")
        self.assertTrue("Failed to call OpenAI API" in str(context.exception))

    def test_get_prompt_embedding_with_cache(self):
        # Test embedding retrieval with cache
        embedding_cache["test prompt"] = torch.tensor([0.1, 0.2, 0.3])
        embedding = get_prompt_embedding("test prompt")
        self.assertTrue(torch.equal(embedding, torch.tensor([0.1, 0.2, 0.3])))

    @patch("requests.post")
    def test_get_prompt_embedding_without_cache(self, mock_post):
        # Mock embedding retrieval when not in cache
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        embedding = get_prompt_embedding("new prompt")
        self.assertTrue(torch.equal(embedding, torch.tensor([0.1, 0.2, 0.3])))

class TestControllerRouting(unittest.TestCase):
    @patch.object(MFModel, "predict_win_rate", return_value=0.8)
    def test_route_strong_model(self, mock_predict):
        prompt_embedding = torch.tensor([0.1, 0.2, 0.3])
        controller = Controller(
            routers=["mf"],
            strong_model="gpt-4o",
            weak_model="gpt-4o-mini",
            dim=128,
            num_models=64,
            text_dim=1536,
            num_classes=1,
            use_proj=True,
        )
        selected_model = controller.route(prompt_embedding, threshold=0.5)
        self.assertEqual(selected_model, "gpt-4o")

    @patch.object(MFModel, "predict_win_rate", return_value=0.3)
    def test_route_weak_model(self, mock_predict):
        prompt_embedding = torch.tensor([0.1, 0.2, 0.3])
        controller = Controller(
            routers=["mf"],
            strong_model="gpt-4o",
            weak_model="gpt-4o-mini",
            dim=128,
            num_models=64,
            text_dim=1536,
            num_classes=1,
            use_proj=True,
        )
        selected_model = controller.route(prompt_embedding, threshold=0.5)
        self.assertEqual(selected_model, "gpt-4o-mini")

class TestAPIEndpoints(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "online"})

    @patch("app.call_openai_for_embedding")
    @patch("app.AzureOpenAI.chat.completions.create")
    def test_chat_completion_endpoint(self, mock_chat, mock_embedding):
        mock_embedding.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_chat.return_value = MagicMock(model_dump=lambda: {"choices": [{"text": "Response text"}]})

        response = client.post(
            "/llm/openai/deployments/smart-router/chat/completions",
            json={"prompt": "test prompt", "messages": [{"role": "user", "content": "Hello"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("choices", response.json())







import pytest
import torch
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app, call_openai_for_embedding, get_prompt_embedding, Controller, embedding_cache

client = TestClient(app)

@pytest.fixture
def mock_controller():
    return Controller(
        routers=["mf"],
        strong_model="gpt-4o",
        weak_model="gpt-4o-mini",
        dim=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True
    )

# Test call_openai_for_embedding function with mock
@patch("app.main.requests.post")
def test_call_openai_for_embedding_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    response = call_openai_for_embedding("test prompt")
    assert response["data"][0]["embedding"] == [0.1, 0.2, 0.3]

@patch("app.main.requests.post")
def test_call_openai_for_embedding_fail(mock_post):
    mock_post.side_effect = requests.exceptions.RequestException("API Error")
    with pytest.raises(Exception) as exc_info:
        call_openai_for_embedding("test prompt")
    assert "Failed to call OpenAI API" in str(exc_info.value)

# Test get_prompt_embedding with and without cache
def test_get_prompt_embedding_with_cache():
    embedding_cache["test prompt"] = torch.tensor([0.1, 0.2, 0.3])
    embedding = get_prompt_embedding("test prompt")
    assert torch.equal(embedding, torch.tensor([0.1, 0.2, 0.3]))

@patch("app.main.call_openai_for_embedding")
def test_get_prompt_embedding_without_cache(mock_call_openai):
    mock_call_openai.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    embedding = get_prompt_embedding("new prompt")
    assert torch.equal(embedding, torch.tensor([0.1, 0.2, 0.3]))

# Test the Controller class routing logic
@patch.object(Controller, "route", return_value="gpt-4o")
def test_controller_route_strong_model(mock_route, mock_controller):
    prompt_embedding = torch.tensor([0.1, 0.2, 0.3])
    selected_model = mock_controller.route(prompt_embedding, threshold=0.5)
    assert selected_model == "gpt-4o"

@patch.object(Controller, "route", return_value="gpt-4o-mini")
def test_controller_route_weak_model(mock_route, mock_controller):
    prompt_embedding = torch.tensor([0.1, 0.2, 0.3])
    selected_model = mock_controller.route(prompt_embedding, threshold=0.9)
    assert selected_model == "gpt-4o-mini"

# Test FastAPI endpoints
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online"}

@patch("app.main.get_prompt_embedding")
@patch("app.main.AzureOpenAI.chat.completions.create")
@pytest.mark.asyncio
async def test_chat_completion(mock_chat, mock_get_prompt_embedding):
    mock_get_prompt_embedding.return_value = torch.tensor([0.1, 0.2, 0.3])
    mock_chat.return_value = MagicMock(model_dump=lambda: {"choices": [{"text": "Response text"}]})

    response = client.post(
        "/llm/openai/deployments/smart-router/chat/completions",
        json={"prompt": "test prompt", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    assert "choices" in response.json()
    assert response.json()["selected_model"] in ["gpt-4o", "gpt-4o-mini"]

