
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from apgenai.embeddingmodelslib.create_index import create_index
from apgenai.embeddingmodelslib.embedding_model import generate_embedding, generate_id
from apgenai.embeddingmodelslib.utils import text_splitter, token_count
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

# Constants and Global Variables
model_token_limit = {"ada": 8191, "all-mpnet": 256, "bge-base": 256}
modal_name_dimension = {"ada": 1536, "all-mpnet": 768, "bge-base": 768}
json_file_tag = ".json"
tables_folder_path = None
table_path_log = None
summary_log_df = None
azure_search_client = None
current_index_scheme = None

# Helper Functions
def auto_create_index(
    directory_path: str,
    azure_endpoint_key: str,
    azure_api_key: str,
    azure_index_name: str,
    model_name: str,
    embedding_field: List[str],
):
    fields_name = ["id"]
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(json_file_tag):
                with open(os.path.join(root, file_name), "r") as f:
                    json_data = json.load(f)
                for field in embedding_field:
                    fields_name.extend([field, f"{field}Vector"])
                fields_name = list(set(fields_name + list(json_data[0].keys())))
                metadata_fields = list(json_data[0].get("metadata", {}).keys())
                create_index(
                    azure_endpoint_key,
                    azure_api_key,
                    azure_index_name,
                    modal_name_dimension[model_name],
                    fields_name,
                    metadata_fields,
                )
                return

def index_schema_fields(field):
    field_dict = {"name": field.name, "type": field.type}
    if field.type == "Edm.ComplexType":
        field_dict["fields"] = [index_schema_fields(sub_field) for sub_field in field.fields]
    else:
        field_dict["fields"] = None
    return field_dict

def _check_index(
    directory_path: str,
    azure_endpoint_key: str,
    azure_api_key: str,
    azure_index_name: str,
    index_fields_schema: Optional[str],
    model_name: str,
    embedding_field: Optional[List[str]],
):
    client = SearchIndexClient(azure_endpoint_key, AzureKeyCredential(azure_api_key))
    try:
        index = client.get_index(azure_index_name)
        return [index_schema_fields(field) for field in index.fields]
    except ResourceNotFoundError:
        if index_fields_schema:
            create_index(
                azure_endpoint_key,
                azure_api_key,
                azure_index_name,
                modal_name_dimension[model_name],
                index_fields_schema=index_fields_schema,
            )
        else:
            auto_create_index(
                directory_path, azure_endpoint_key, azure_api_key, azure_index_name, model_name, embedding_field
            )
        index = client.get_index(azure_index_name)
        return [index_schema_fields(field) for field in index.fields]
    except HttpResponseError as e:
        logging.error("Invalid credentials: %s", e)
        raise

def check_token_limit(json_data: Dict, embedding_field: List[str], chunk_size: int) -> List[Dict]:
    updated_json_data = []
    for record in json_data:
        flag = False
        for key, value in record.items():
            if value and (key in embedding_field or key == "page_content") and token_count(str(value)) > chunk_size:
                flag = True
                chunks = text_splitter(str(value), chunk_size)
                for chunk in chunks:
                    new_record = record.copy()
                    new_record[key] = chunk
                    updated_json_data.append(new_record)
        if not flag:
            updated_json_data.append(record)
    return updated_json_data

def table_parquet(json_data: Dict):
    global tables_folder_path, table_path_log
    sub_folder_path = os.path.join(tables_folder_path, f"page_no_{json_data['metadata']['page_no']}")
    os.makedirs(sub_folder_path, exist_ok=True)
    for table in json_data["page_tables"]:
        columns = [str(element) for element in table[1]]
        df = pd.DataFrame(table[2:], columns=columns)
        file_name = os.path.join(sub_folder_path, f"{table[0][0].replace('/', ' ')}.parquet")
        df.to_parquet(file_name)

def json_embedding(
    json_data: Any,
    model_name: str,
    endpoint_key: str,
    api_key: str,
    deployment_id: str,
    embedding_field: List[str],
    chunk_size: int,
    extract_table_as_parquet: bool,
) -> List[Dict]:
    global current_index_scheme
    json_data = check_token_limit(json_data, embedding_field, chunk_size)
    embeddings = []
    for record in json_data:
        processed_record = {
            "id": generate_id(),
            **record,
            **{
                f"{field}Vector": generate_embedding([str(record[field])], model_name, endpoint_key, api_key, deployment_id)
                for field in embedding_field if field in record
            },
        }
        if extract_table_as_parquet and "page_tables" in record:
            table_parquet(record)
        embeddings.append(processed_record)
    return embeddings

def load_index(json_data: List[Dict], azure_index_name: str):
    global azure_search_client
    try:
        filter_query = f"metadata/source eq '{json_data[0]['metadata']['source']}'"
        existing_docs = azure_search_client.search(search_text="*", filter=filter_query)
        for doc in existing_docs:
            azure_search_client.delete_documents(documents=[{"id": doc["id"]}])
        azure_search_client.upload_documents(documents=json_data)
    except Exception as e:
        logging.error("Failed to load documents: %s", e)
        raise

# Main Embedding Pipeline
def dir_embedding(
    path: str,
    model_name: str,
    model_endpoint: str,
    model_api_key: str,
    deployment_id: str,
    azure_endpoint_key: str,
    azure_api_key: str,
    azure_index_name: str,
    embedding_field: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    summary_log_folder_path: Optional[str] = None,
    index_fields_schema: Optional[str] = None,
    extract_table_as_parquet: bool = False,
    **kwargs,
):
    global summary_log_df, azure_search_client, tables_folder_path
    summary_log_df = pd.DataFrame(columns=["Timestamp", "FileName", "Status", "Message"])

    try:
        if model_name not in model_token_limit:
            raise ValueError("Invalid model name.")

        azure_search_client = SearchClient(
            azure_endpoint_key,
            azure_index_name,
            AzureKeyCredential(azure_api_key),
        )

        tables_folder_path = os.path.join(path, "tables")
        os.makedirs(tables_folder_path, exist_ok=True)

        _check_index(
            path,
            azure_endpoint_key,
            azure_api_key,
            azure_index_name,
            index_fields_schema,
            model_name,
            embedding_field,
        )

        if os.path.isfile(path):
            with open(path, "r") as f:
                json_data = json.load(f)
            embeddings = json_embedding(
                json_data, model_name, model_endpoint, model_api_key, deployment_id, embedding_field, chunk_size, extract_table_as_parquet
            )
            load_index(embeddings, azure_index_name)
        elif os.path.isdir(path):
            for file_name in os.listdir(path):
                if file_name.endswith(json_file_tag):
                    with open(os.path.join(path, file_name), "r") as f:
                        json_data = json.load(f)
                    embeddings = json_embedding(
                        json_data, model_name, model_endpoint, model_api_key, deployment_id, embedding_field, chunk_size, extract_table_as_parquet
                    )
                    load_index(embeddings, azure_index_name)
    except Exception as e:
        logging.error("Embedding process failed: %s", e)
