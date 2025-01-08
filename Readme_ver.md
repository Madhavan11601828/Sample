# Documentation for the Code

## Overview

This code integrates Azure Form Recognizer, Azure OpenAI, and Azure Cognitive Search services. It processes data from input files, extracts text using Azure Form Recognizer, generates embeddings with Azure OpenAI models, stores embeddings in Azure Cognitive Search, and uses them to answer queries through a language model.

---

## Prerequisites

Ensure you have the following dependencies installed:

- `apgenai-embeddingmodelslib==0.1.0.dev6`
- `apgenai-datapreprocessing==1.0.1`
- `openai==1.11.1`

Install them using:

```python
%pip install apgenai-embeddingmodelslib==0.1.0.dev6 apgenai-datapreprocessing==1.0.1 openai==1.11.1
```

---

## Key Steps in the Code

### 1. Import Necessary Libraries

```python
import logging
from apgenai.datapreprocessing.formrecognizer import extract_text_via_formrec_in_dir
from apgenai.embeddingmodelslib.read_load_json import dir_embedding
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
```

- **Logging Configuration**:
  - Errors from `py4j.java_gateway` are set to `ERROR` level.
  - The basic logging level is set to `INFO`.

### 2. Extract Text Using Azure Form Recognizer

The function `extract_text_via_formrec_in_dir` extracts text from input files in a directory and saves it as JSON files.

```python
json_data = extract_text_via_formrec_in_dir(
    input_dir=FORM_RECOGNISER_INPUT_DIR,
    endpoint=FORM_RECOGNISER_ENDPOINT,
    key=FORM_RECOGNISER_KEY,
    output_dir=FORM_RECOGNISER_OUTPUT_DIR,
    model_id=FORM_RECOGNISER_MODEL_ID,
    locale=FORM_RECOGNISER_LOCALE,
    summary_logging_path=SUMMARY_LOGGING_PATH
)
```

### 3. Generate Embeddings Using Azure OpenAI

The `dir_embedding` function generates embeddings for the JSON files and uploads them to Azure Cognitive Search.

```python
dir_embedding(
    dir_path=EMBEDDING_INPUT_JSON_DIR,
    model_name=EMBEDDING_MODEL_NAME,
    model_endpoint=EMBEDDING_MODEL_ENDPOINT,
    model_api_key=EMBEDDING_MODEL_API_KEY,
    deployment_id=EMBEDDING_MODEL_DEPLOYMENT_ID,
    azure_cognitive_api_key=AZURE_COGNITIVE_SEARCH_KEY,
    azure_cognitive_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
    azure_cognitive_index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
    api_version=EMBEDDING_MODEL_API_VERSION,
    embedding_field=EMBEDDING_FIELD,
    extract_table_as_parquet=extract_table_as_parquet
)
```

### 4. Query Azure Cognitive Search

The code queries Azure Cognitive Search for documents related to a specific query.

```python
credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_KEY)
sclient = SearchClient(
    AZURE_COGNITIVE_SEARCH_ENDPOINT,
    AZURE_COGNITIVE_SEARCH_INDEX_NAME,
    credential
)

query = "Explain about global messaging on the customer experience."
resp = sclient.search(query, top=2)

docs = []
for doc in resp:
    docs.append(doc)

len(docs)
```

### 5. Generate a Query-Response with Azure OpenAI

Azure OpenAI's GPT-4 model is used to answer the query based on the retrieved documents.

```python
aoi = AzureOpenAI(
    azure_endpoint=openai.api_base,
    api_key=openai.api_key,
    azure_deployment="gpt-4"
)

context = "\n====\n".join([doc["page_content"] for doc in docs])

user_prompt = """Answer the query based on the given context:
{context}

Query: {query}
Your answer:"""

messages = [
    {"role": "system", "content": "You're a helpful assistant. Answer the user queries"},
    {"role": "user", "content": user_prompt.format(query=query, context=context)},
]

resp = aoi.chat.completions.create(
    model="gpt-4",
    messages=messages
)

print(resp.choices[0].message.content)
```

---

## Key Variables

| Variable Name                       | Description                                           |
|-------------------------------------|-------------------------------------------------------|
| `FORM_RECOGNISER_INPUT_DIR`         | Directory of input files for Form Recognizer.         |
| `FORM_RECOGNISER_ENDPOINT`          | Azure Form Recognizer endpoint.                      |
| `FORM_RECOGNISER_KEY`               | Azure Form Recognizer API key.                       |
| `FORM_RECOGNISER_OUTPUT_DIR`        | Directory to save extracted JSON files.              |
| `FORM_RECOGNISER_MODEL_ID`          | Form Recognizer model ID.                            |
| `FORM_RECOGNISER_LOCALE`            | Locale for Form Recognizer.                          |
| `SUMMARY_LOGGING_PATH`              | Path for logging summary of operations.              |
| `EMBEDDING_INPUT_JSON_DIR`          | Directory of JSON files for embeddings.              |
| `EMBEDDING_MODEL_NAME`              | Name of the embedding model.                         |
| `EMBEDDING_MODEL_ENDPOINT`          | Azure OpenAI model endpoint.                         |
| `EMBEDDING_MODEL_API_KEY`           | API key for Azure OpenAI.                            |
| `EMBEDDING_MODEL_DEPLOYMENT_ID`     | Deployment ID for the embedding model.               |
| `AZURE_COGNITIVE_SEARCH_KEY`        | Key for Azure Cognitive Search.                      |
| `AZURE_COGNITIVE_SEARCH_ENDPOINT`   | Endpoint for Azure Cognitive Search.                 |
| `AZURE_COGNITIVE_SEARCH_INDEX_NAME` | Name of the Azure Cognitive Search index.            |
| `EMBEDDING_MODEL_API_VERSION`       | API version for the embedding model.                 |
| `EMBEDDING_FIELD`                   | Field name for storing embeddings.                   |
| `query`                             | Query string for Azure Cognitive Search.             |

---

## Workflow Summary

1. Extract text from input files using Azure Form Recognizer.
2. Generate embeddings for the extracted text using Azure OpenAI models.
3. Store the embeddings in Azure Cognitive Search.
4. Query Azure Cognitive Search for relevant documents.
5. Use Azure OpenAI's GPT-4 model to generate answers based on the retrieved documents.

---

## Notes

- Ensure that all Azure resources (Form Recognizer, Cognitive Search, and OpenAI) are properly set up and configured.
- Replace placeholder variables (e.g., `FORM_RECOGNISER_INPUT_DIR`, `EMBEDDING_MODEL_NAME`) with actual values.
- The code assumes you have the necessary permissions and keys for all services.
