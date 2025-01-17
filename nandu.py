import json
import logging
import os
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List, Optional

import pandas as pd
from apgenai.embeddingmodelslib.create_index import create_index
from apgenai.embeddingmodelslib.embedding_model import generate_embedding
from apgenai.embeddingmodelslib.embedding_model import generate_id
from apgenai.embeddingmodelslib.utils import text_splitter
from apgenai.embeddingmodelslib.utils import token_count
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

root_dir = os.path.abspath(os.path.join(__file__, "../"))

error_msg = "Unexpected error: "

model_token_limit = {"ada": 8191, "all-mpnet": 256, "bge-base": 256}

modal_name_dimension = {"ada": 1536, "all-mpnet": 768, "bge-base": 768}


sub_folder_path = None
tables_folder_path = ""
table_path_log = ""

summary_file_name = ""
azure_search_client = None
file_path = None
json_file_tag = ".json"


# Configuration class for model-related parameters
class ModelConfig:
    def __init__(self, model_name: str, model_endpoint: str, model_api_key: str, deployment_id: str, api_version: Optional[str] = None):
        self.model_name = model_name
        self.model_endpoint = model_endpoint
        self.model_api_key = model_api_key
        self.deployment_id = deployment_id
        self.api_version = api_version


# Configuration class for Azure Cognitive Search-related parameters
class AzureConfig:
    def __init__(self, endpoint_key: str, api_key: str, index_name: str):
        self.endpoint_key = endpoint_key
        self.api_key = api_key
        self.index_name = index_name


# Configuration class for embedding generation-related parameters
class EmbeddingConfig:
    def __init__(self, embedding_field: Optional[List[str]] = None, chunk_size: Optional[int] = None, extract_table_as_parquet: bool = False):
        self.embedding_field = embedding_field
        self.chunk_size = chunk_size
        self.extract_table_as_parquet = extract_table_as_parquet


# Configuration class for logging and index schema-related parameters
class LogConfig:
    def __init__(self, summary_log_folder_path: Optional[str] = None, index_fields_schema: Optional[str] = None):
        self.summary_log_folder_path = summary_log_folder_path
        self.index_fields_schema = index_fields_schema


def auto_create_index(
    directory_path: str,
    azure_cognitve_endpoint_key: str,
    azure_cognitve_api_key: str,
    azure_cognitve_index_name: str,
    model_name: str,
    embedding_field: List = None,
):
    fields_name = ["id"]
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(json_file_tag):
                first_filepath = os.path.join(root, file_name)
                with open(first_filepath, "r") as f:
                    json_data = json.load(f)
                for i in embedding_field:
                    fields_name.append(i)
                    fields_name.append(i + "Vector")
                fields_name = set(fields_name + list(json_data[0].keys()))
                fields_name = list(fields_name)
                metadata_field_name = list(json_data[0]["metadata"].keys())
                create_index(
                    azure_cognitve_endpoint_key,
                    azure_cognitve_api_key,
                    azure_cognitve_index_name,
                    modal_name_dimension[model_name],
                    fields_name,
                    metadata_field_name,
                )
                return None


def index_schema_fields(field):
    field_dict = {
        "name": field.name,
        "type": field.type,
    }

    if field.type == "Edm.ComplexType":
        # Recursively handle sub-fields for complex fields
        field_dict["fields"] = [index_schema_fields(sub_field) for sub_field in field.fields]
    else:
        field_dict["fields"] = None
    return field_dict


def _check_index(
    directory_path: str,
    azure_cognitve_endpoint_key: str,
    azure_cognitve_api_key: str,
    azure_cognitve_index_name: str,
    index_fields_schema: str,
    model_name: str,
    embedding_field: List = None,
):
    client = SearchIndexClient(azure_cognitve_endpoint_key, AzureKeyCredential(azure_cognitve_api_key))
    try:
        index = client.get_index(azure_cognitve_index_name)
        return [index_schema_fields(field) for field in index.fields]
    # Thorws the ResourceNotFoundError error when the index is not created,
    except ResourceNotFoundError:
        if index_fields_schema is not None:
            create_index(
                endpoint_key=azure_cognitve_endpoint_key,
                api_key=azure_cognitve_api_key,
                index_name=azure_cognitve_index_name,
                index_fields_schema=index_fields_schema,
                vector_search_dimensions=modal_name_dimension[model_name],
            )
        else:
            # create the index using first file
            auto_create_index(
                directory_path,
                azure_cognitve_endpoint_key,
                azure_cognitve_api_key,
                azure_cognitve_index_name,
                model_name,
                embedding_field,
            )
        index = client.get_index(azure_cognitve_index_name)
        return [index_schema_fields(field) for field in index.fields]
    except HttpResponseError:
        logging.error("Invalid credentional, please check")
        raise


def check_token_limit(json_data: Dict, embedding_field: List, chunk_size: int) -> List:
    """
    check_token_limit - Splits the json_data key field into chunks according to model token limit.

    Args:
        json_data (Dict): Json Data
        embedding_field (List): filed name for generating the emebddings

    Returns:
        List: A list with key filed split into chunks
    """
    flag = False
    updated_json_data = []
    for data in json_data:
        for key, value in data.items():
            token_size = token_count(str(value))
            if value and (key in embedding_field or key in ["page_content"]) and (token_size > chunk_size):
                flag = True
                chunk = text_splitter(str(value), chunk_size)
                for i in chunk:
                    chunk_dict = data.copy()
                    chunk_dict[key] = i
                    updated_json_data.append(chunk_dict)
        if flag is False:
            updated_json_data.append(data)
    return updated_json_data


def table_parquet(json_data: Dict):
    """
    table_parquet - Extract the page_tables field and save as parquet file.

    Args:
        json_data (Dict): json data
    """
    global tables_folder_path
    global root_path
    global table_path_log
    sub_folder_path = os.path.join(tables_folder_path, os.path.relpath(root_path, os.path.dirname(tables_folder_path)))
    sub_folder_path = os.path.join(sub_folder_path, str(summary_file_name).split(".")[0])
    table_path_log = sub_folder_path
    os.makedirs(sub_folder_path, exist_ok=True)
    sub_folder = os.path.join(sub_folder_path, "page_no_" + str(json_data["metadata"]["page_no"]))
    os.makedirs(sub_folder, exist_ok=True)
    value = json_data["page_tables"]
    try:
        for table in value:
            column = [str(element) for element in table[1]]
            df = pd.DataFrame(table[2:], columns=column)
            sub_table_name = str(table[0][0]).replace("/", " ")
            table_folder = os.path.join(sub_folder, sub_table_name)
            os.makedirs(table_folder, exist_ok=True)
            table_file_name = os.path.join(table_folder, sub_table_name + ".parquet")
            df.to_parquet(table_file_name)
    except Exception as e:
        table_path_log = "Unable to save the table as parquet due to " + e
        logging.error(f"Unable to save the table due to {e}")


def validate_and_adjust_data(data, field_map):
    """Validate and adjust data based on the field map.

    Args:
        data (dict): The data to validate and adjust.
        field_map (list): The field map describing expected fields and their types.

    Returns:
        dict: The validated and adjusted data.

    Raises:
        ValueError: If a field value does not meet the expected type requirements.
    """
    for field in field_map:
        field_name = field.get("name")
        if data and field_name in data:
            data[field_name] = validate_field(data[field_name], field)
    return data


def validate_field(value, field):
    """Validate and adjust a single field's value.

    Args:
        value: The value to validate and adjust.
        field (dict): The field description from the field map.

    Returns:
        The validated and adjusted value.

    Raises:
        ValueError: If the value does not meet the expected type requirements.
    """
    if value is None:
        return None
    elif field.get("type") == "Edm.ComplexType" and field.get("fields"):
        # Recursively validate and adjust nested fields
        return validate_and_adjust_data(value, field["fields"])
    elif "Collection(Edm." in field.get("type"):
        # Ensure the field is a list
        if not isinstance(value, list):
            raise ValueError(f"{field['name']} must be a list of strings")
    elif field.get("type") == "Edm.String":
        # Ensure the field is a string
        if not isinstance(value, str):
            return json.dumps(value)
    return value


# generate the embedding for field
def _json_data_processing(
    data: Any,
    model_name: str,
    endpoint_key: str,
    api_key: str,
    deployment_id: str,
    embedding_field: List = None,
    api_version: str = None,
    extract_table_as_parquet: bool = False,
) -> Dict:
    """
    _json_data_processing - Read json data and return List[Dict] that contains the embeddings and string data of key fields using given model_name

    Parameters :

    filepath_or_json_data (Any): json file path or json_data according to user requirement
    model_name (str): Name of the model(e.g ada,all-mpnet,bge-base)
    endpoint_key (str) : Azure Openai Endpoint key
    api_key (str) : Azure Openai API key
    deployment_id(str) : Name of deployed of model(e.g bge-base-v2,t5-large-13)
    api_version (str)(Optional) : The API version follows the YYYY-MM-DD format.(eg.2023-03-15-preview,2022-12-01) needed for openai model
    embedding_filed (list)(Optional) : Optional filed name for generating the emebddings
    extract_table_as_parquet (bool) : if you want to save the page_tables fields then True else False
    Returns:
            List[Embeddings]: A list conatins embeddings and string data.

    """
    emd_dictionary = {}
    data = validate_and_adjust_data(data, current_index_scheme)
    for key, value in data.items():
        emd_dictionary["id"] = generate_id()
        if key == "page_tables" and value and extract_table_as_parquet:
            table_parquet(data)
        elif value and (key in embedding_field or key in ["page_content"]):
            if key == "metadata":
                value = {key: value for key, value in data["metadata"].items()}
                value = {key: value for key, value in data["metadata"].items()}
                emd_dictionary[key] = value
            else:
                emd_dictionary[key] = value
            emd_dictionary[f"{key}Vector"] = generate_embedding(
                [str(value)], model_name, endpoint_key, api_key, deployment_id, api_version
            )
        elif key == "metadata":
            value = {key: value for key, value in data["metadata"].items()}
            value = {key: value for key, value in data["metadata"].items()}
            emd_dictionary[key] = value
        elif key != "page_tables":
            emd_dictionary[key] = value
    return emd_dictionary


# summary log function to save the log as parquet
def _summary_log(dir_path: str):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(dir_path, timestamp)
        os.makedirs(folder_path, exist_ok=True)
        # corr
        summary_log_df.to_parquet(os.path.join(folder_path, timestamp + ".parquet"))
        status_counts = summary_log_df["STATUS"].value_counts()
        print("Count of files that have successfully processed and those facing failures in the embedding generation.")
        print(status_counts.to_string())
        print("The following is the file path for the tables folder in the form of a parquet file.", tables_folder_path)
        print("The following is the file path for the summary log folder in the form of a parquet file.", folder_path)
        print(summary_log_df)
    except Exception as e:
        logging.error(f"Unable to save the summary log causing Exception {e}")


def __dataframe_info(
    model_name: str,
    model_endpoint: str,
    model_api_key: str,
    deployment_id: str,
    azure_cognitve_index_name: str,
    dataframe_files_info: pd.DataFrame = None,
    embedding_field: List = None,
    chunk_size: int = None,
    api_version: str = None,
    extract_table_as_parquet: bool = False,
) -> List:
    """
    __dataframe_info - accept the dataframe which contains the json files path and generate embeddings and load into azure cognitve search, saved tables as parquet.
    Parameters :

    model_name (str): Name of the model(e.g ada,all-mpnet,bge-base)
    model_endpoint (str) : Model endpoint
    model_api_key (str) : Model api key
    deployment_id(str) : Name of deployed of model(e.g bge-base-v2,t5-large-13
    azure_cognitve_index_name : Index name for azure cognitve search
    embedding_filed (List)(Optional) : Optional filed name for generating the emebddings
    chunk_size (int)(Optional) : Chunk size(in terms of token) to split the text
    api_version (str)(Optional) : The API version follows the YYYY-MM-DD format.(eg.2023-03-15-preview,2022-12-01) needed for openai model
    extract_table_as_parquet (bool) : if you want to save the page_tables fields then True else False
    """
    for _, row in dataframe_files_info.iterrows():
        if row["TYPE"] == "FILE":
            file = os.path.join(row["PATH"], row["NAME"])
            file_name = row["NAME"]
            try:
                if file_name.endswith(json_file_tag):
                    global summary_file_name
                    summary_file_name = file_name
                    global root_path
                    root_path = file
                    global file_path
                    file_path = file
                    json_data = json_embedding(
                        file_path,
                        model_name,
                        model_endpoint,
                        model_api_key,
                        deployment_id,
                        embedding_field,
                        chunk_size,
                        api_version,
                        extract_table_as_parquet,
                    )
                    load_index(
                        json_data,
                        azure_cognitve_index_name,
                    )

                    # execution log of file
                    last_row = summary_log_df.iloc[-1]
                    file_log = "\n".join(last_row.astype(str))
                    print(file_log)
            except Exception as e:
                summary_log_df.loc[summary_log_df.shape[0]] = {
                    "EXECUTION_TIMESTAMP_UTC": str(datetime.utcnow()),
                    "FILE_NAME": summary_file_name,
                    "FILE_PATH": file_path,
                    "TABLE_PATH": table_path_log,
                    "STATUS": "FAILED",
                    "SUCCESS_OR_ERROR_MESSAGE": str(f"Unable to load the file {file_name} causing Exception {e}"),
                }
                logging.error(f"Unable to load the file {file_name} causing Exception {e}")
                continue


def single_file_embedding(
    file_path: str,
    model_name: str,
    model_endpoint: str,
    model_api_key: str,
    deployment_id: str,
    azure_cognitve_index_name: str,
    embedding_field: List = None,
    chunk_size: int = None,
    api_version: str = None,
    extract_table_as_parquet: bool = False,
):
    """
    single_file_embedding - accept the file path of a JSON file, generate embeddings, and load into Azure Cognitive Search, optionally save tables as Parquet.
    Parameters :

    file_path (str): Path to the JSON file.
    model_name (str): Name of the model (e.g., ada, all-mpnet, bge-base).
    model_endpoint (str): Model endpoint.
    model_api_key (str): Model API key.
    deployment_id (str): Name of the deployed model (e.g., bge-base-v2, t5-large-13).
    azure_cognitve_index_name (str): Index name for Azure Cognitive Search.
    embedding_field (List, optional): Optional field names for generating embeddings.
    chunk_size (int, optional): Chunk size (in terms of tokens) to split the text.
    api_version (str, optional): API version (e.g., 2023-03-15-preview, 2022-12-01).
    extract_table_as_parquet (bool): Whether to save the page tables as Parquet files.
    """
    # Initialize summary_log_df and table_path_log
    global summary_log_df
    if 'summary_log_df' not in globals():
        summary_log_df = pd.DataFrame(
            columns=[
                "EXECUTION_TIMESTAMP_UTC",
                "FILE_NAME",
                "FILE_PATH",
                "TABLE_PATH",
                "STATUS",
                "SUCCESS_OR_ERROR_MESSAGE",
            ]
        )
    table_path_log = None

    try:
        json_data = json_embedding(
            file_path,
            model_name,
            model_endpoint,
            model_api_key,
            deployment_id,
            embedding_field,
            chunk_size,
            api_version,
            extract_table_as_parquet,
        )
        load_index(
            json_data,
            azure_cognitve_index_name,
        )
        # Execution log of file
        last_row = summary_log_df.iloc[-1]
        file_log = "\n".join(last_row.astype(str))
        print(file_log)
    except Exception as e:
        summary_log_df.loc[summary_log_df.shape[0]] = {
            "EXECUTION_TIMESTAMP_UTC": str(datetime.utcnow()),
            "FILE_NAME": os.path.basename(file_path),
            "FILE_PATH": file_path,
            "TABLE_PATH": table_path_log,
            "STATUS": "FAILED",
            "SUCCESS_OR_ERROR_MESSAGE": str(f"Unable to load the file {file_path} causing Exception {e}"),
        }
        logging.error(f"Unable to load the file {file_path} causing Exception {e}")


def __dir_embedding(
    dir_path: str,
    model_name: str,
    model_endpoint: str,
    model_api_key: str,
    deployment_id: str,
    azure_cognitve_index_name: str,
    embedding_field: List = None,
    chunk_size: int = None,
    api_version: str = None,
    extract_table_as_parquet: bool = False,
):
    """
    dir_embedding - accept the directory path which contains the json files and generate embeddings and load into azure cognitve search, saved tables as parquet.
    Parameters :

    dir_path (str): Directory path.
    model_name (str): Name of the model(e.g ada,all-mpnet,bge-base).
    model_endpoint (str) : Model endpoint.
    model_api_key (str) : Model api key.
    deployment_id(str) : Name of deployed of model(e.g bge-base-v2,t5-large-13).
    azure_cognitve_index_name : Index name for azure cognitve search.
    embedding_filed (List)(Optional) : Optional filed name for generating the emebddings.
    chunk_size (int)(Optional) : Chunk size(in terms of token) to split the text.
    api_version (str)(Optional) : The API version follows the YYYY-MM-DD format.(eg.2023-03-15-preview,2022-12-01) needed for openai model.
    extract_table_as_parquet (bool) : if you want to save the page_tables fields then True else False.
    """
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            try:
                if file_name.endswith(json_file_tag):
                    global summary_file_name
                    summary_file_name = file_name
                    global root_path
                    root_path = root
                    global file_path
                    file_path = os.path.join(root, file_name)
                    json_data = json_embedding(
                        file_path,
                        model_name,
                        model_endpoint,
                        model_api_key,
                        deployment_id,
                        embedding_field,
                        chunk_size,
                        api_version,
                        extract_table_as_parquet,
                    )
                    load_index(
                        json_data,
                        azure_cognitve_index_name,
                    )
                    # execution log of file
                    last_row = summary_log_df.iloc[-1]
                    file_log = "\n".join(last_row.astype(str))
                    print(file_log)
            except Exception as e:
                summary_log_df.loc[summary_log_df.shape[0]] = {
                    "EXECUTION_TIMESTAMP_UTC": str(datetime.utcnow()),
                    "FILE_NAME": summary_file_name,
                    "FILE_PATH": file_path,
                    "TABLE_PATH": table_path_log,
                    "STATUS": "FAILED",
                    "SUCCESS_OR_ERROR_MESSAGE": str(f"Unable to load the file {file_name} causing Exception {e}"),
                }
                logging.error(f"Unable to load the file {file_name} causing Exception {e}")
                continue


# Main function to process the directory or file to generate and load embeddings
def dir_embedding(
    path: str,
    model_config: ModelConfig,
    azure_config: AzureConfig,
    embedding_config: EmbeddingConfig,
    log_config: LogConfig,
    dataframe_files_info: Optional[pd.DataFrame] = None,
) -> List:
    """
    dir_embedding - accept the directory path or file path, generate embeddings, and load into Azure Cognitive Search, save tables as Parquet.
    Parameters :

    path (str): Directory path or file path.
    model_config (ModelConfig): Configuration for the model.
    azure_config (AzureConfig): Configuration for Azure Cognitive Search.
    embedding_config (EmbeddingConfig): Configuration for embedding generation.
    log_config (LogConfig): Configuration for logging and index schema.
    dataframe_files_info (pd.DataFrame, optional): DataFrame containing file information for generating embeddings.
    """
    
    # Validate if the provided path is a valid directory or file path
    if not os.path.exists(path):
        raise ValueError("The provided path does not exist.")
    
    # Initialize the summary log DataFrame
    global summary_log_df
    summary_log_df = pd.DataFrame(
        columns=[
            "EXECUTION_TIMESTAMP_UTC",
            "FILE_NAME",
            "FILE_PATH",
            "TABLE_PATH",
            "STATUS",
            "SUCCESS_OR_ERROR_MESSAGE",
        ]
    )

    try:
        logging.info("Starting the embedding generation process.")

        # Validate the model name
        if model_config.model_name not in ["ada", "all-mpnet", "bge-base"]:
            logging.error(
                "Model is Invalid, please check model name. Here is the list of supported embedding models: [text-embedding-ada, all-mpnet, bge-base]"
            )
            raise RuntimeError
        else:
            # Generate a sample embedding to validate the model and API
            sample_string = "api_check"
            generate_embedding([sample_string], model_config.model_name, model_config.model_endpoint, model_config.model_api_key, model_config.deployment_id, model_config.api_version)

        # Set the chunk size based on the provided value or the model's token limit
        if embedding_config.chunk_size is not None:
            chunk_size = min(embedding_config.chunk_size, model_token_limit[model_config.model_name] // 2)
        else:
            chunk_size = model_token_limit[model_config.model_name] // 2

        # Create the summary log directory
        if log_config.summary_log_folder_path is not None:
            summary_log_folder_path = os.path.join(log_config.summary_log_folder_path, "embedding_generation_log")
            os.makedirs(summary_log_folder_path, exist_ok=True)
        else:
            summary_log_folder_path = os.path.join(path, "embedding_generation_log")
            os.makedirs(summary_log_folder_path, exist_ok=True)

        # Initialize the Azure Search client
        global azure_search_client
        azure_search_client = SearchClient(
            azure_config.endpoint_key,
            azure_config.index_name,
            AzureKeyCredential(azure_config.api_key),
        )

        # Create the tables directory
        global tables_folder_path
        tables_folder_path = os.path.join(path, "tables")
        os.makedirs(tables_folder_path, exist_ok=True)

        # Check and create the index if necessary
        global current_index_scheme
        current_index_scheme = _check_index(
            path,
            azure_config.endpoint_key,
            azure_config.api_key,
            azure_config.index_name,
            log_config.index_fields_schema,
            model_config.model_name,
            embedding_config.embedding_field,
        )

        # Process files based on the provided DataFrame or directory/file path
        if dataframe_files_info is not None:
            __dataframe_info(
                model_config.model_name,
                model_config.model_endpoint,
                model_config.model_api_key,
                model_config.deployment_id,
                azure_config.index_name,
                dataframe_files_info,
                embedding_config.embedding_field,
                chunk_size,
                model_config.api_version,
                embedding_config.extract_table_as_parquet,
            )
        else:
            if os.path.isdir(path):
                # Process directory
                __dir_embedding(
                    path,
                    model_config.model_name,
                    model_config.model_endpoint,
                    model_config.model_api_key,
                    model_config.deployment_id,
                    azure_config.index_name,
                    embedding_config.embedding_field,
                    chunk_size,
                    model_config.api_version,
                    embedding_config.extract_table_as_parquet,
                )
            elif os.path.isfile(path):
                # Process single file
                single_file_embedding(
                    path,
                    model_config.model_name,
                    model_config.model_endpoint,
                    model_config.model_api_key,
                    model_config.deployment_id,
                    azure_config.index_name,
                    embedding_config.embedding_field,
                    chunk_size,
                    model_config.api_version,
                    embedding_config.extract_table_as_parquet,
                )
            else:
                raise ValueError("The provided path is neither a directory nor a file.")

        logging.info("Completed the embedding generation process.")
        
    except Exception as e:
        logging.error(f"An error occurred during the embedding generation process: {e}")
        raise

    # Log the summary of the execution
    _summary_log(summary_log_folder_path)


# json data into embedding
def json_embedding(
    filepath_or_json_data: Any,
    model_name: str,
    endpoint_key: str,
    api_key: str,
    deployment_id: str,
    embedding_field: List = None,
    chunk_size: int = None,
    api_version: str = None,
    extract_table_as_parquet: bool = False,
) -> List:
    """
    json_generate_embedding - Read json file or json_data and return List[Dict] that contains the embeddings and string data of key fields using given model_name.

    Parameters :

    filepath_or_json_data (Any): json file path or json_data according to user requirement
    model_name (str): Name of the model(e.g ada,all-mpnet,bge-base)
    endpoint_key (str) : Azure Openai Endpoint key
    api_key (str) : Azure Openai API key
    deployment_id(str) : Name of deployed of model(e.g bge-base-v2,t5-large-13)
    api_version (str)(Optional) : The API version follows the YYYY-MM-DD format.(eg.2023-03-15-preview,2022-12-01) needed for openai model
    embedding_filed (list)(Optional) : Optional filed name for generating the emebddings
    chunk_size (int)(Optional) : Chunk size(in terms of token) to split the text
    extract_table_as_parquet (bool) : if you want to save the page_tables fields then True else False
    Returns:
            List[Embeddings]: A list conatins embeddings and string data.

    """
    if model_name not in ["ada", "all-mpnet", "bge-base"]:
        logging.error("Model Name is Invalid,please check model name ")
        raise RuntimeError
    else:
        sample_string = "api_check"
        generate_embedding([sample_string], model_name, endpoint_key, api_key, deployment_id, api_version)
    embeddings = []
    try:
        if isinstance(filepath_or_json_data, str):
            with open(filepath_or_json_data, "r") as f:
                json_data = json.load(f)
        else:
            json_data = filepath_or_json_data
        json_data = check_token_limit(json_data, embedding_field, chunk_size)
        for data in json_data:
            emd_dictionary = _json_data_processing(
                data,
                model_name,
                endpoint_key,
                api_key,
                deployment_id,
                embedding_field,
                api_version,
                extract_table_as_parquet,
            )

            embeddings.append(emd_dictionary)
        return embeddings
    except FileNotFoundError as e:
        summary_log_df.loc[summary_log_df.shape[0]] = {
            "EXECUTION_TIMESTAMP_UTC": str(datetime.utcnow()),
            "FILE_NAME": summary_file_name,
            "FILE_PATH": file_path,
            "TABLE_PATH": table_path_log,
            "STATUS": "FAILED",
            "SUCCESS_OR_ERROR_MESSAGE": str(f"File not found due to error - {e}"),
        }
        logging.error(f"File not found{e}")


# load the json data into azure cognitive search
def load_index(
    json_data: List[Dict],
    index_name: str,
):
    """
    load_index - function to Load Json data into azure cognitive search index, if index dosen't exit it will create new one.

    Parameters :
    json_data (List[Dict]) : List of Dictionary content the fields(id,page_content,source,pagecontentVector,sourceVector,pageno)
    index_name (str):  Name of Azure Congnitive Search
    """
    try:
        # if file exits then delete and upload the new one
        file_metadata = json_data[0]["metadata"]["source"]
        filter_cond = f"metadata/source eq '{file_metadata}'"
        results = azure_search_client.search(search_text="*", filter=filter_cond)
        for rec in results:
            azure_search_client.delete_documents(documents=[{"id": rec["id"]}])
        # upload the document
        azure_search_client.upload_documents(documents=json_data)
        logging.info(f"Uploaded {len(json_data)} documents in total in {index_name} Index ")
        summary_log_df.loc[summary_log_df.shape[0]] = {
            "EXECUTION_TIMESTAMP_UTC": str(datetime.utcnow()),
            "FILE_NAME": summary_file_name,
            "FILE_PATH": file_path,
            "TABLE_PATH": table_path_log,
            "STATUS": "SUCCESS",
            "SUCCESS_OR_ERROR_MESSAGE": str(f"Uploaded {len(json_data)} documents in total in {index_name} Index "),
        }
    except Exception as e:
        summary_log_df.loc[summary_log_df.shape[0]] = {
            "EXECUTION_TIMESTAMP_UTC": str(datetime.utcnow()),
            "FILE_NAME": summary_file_name,
            "FILE_PATH": file_path,
            "TABLE_PATH": table_path_log,
            "STATUS": "FAILED",
            "SUCCESS_OR_ERROR_MESSAGE": str(f"Failed due to - {e}"),
        }
        logging.error(f"{e}")
