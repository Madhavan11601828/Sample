
# SharePoint Data Import Library Wiki

## Introduction
The **SharePoint Data Import Library** is a Python-based utility designed to interact with Microsoft SharePoint sites. It allows users to manage files, folders, and metadata in SharePoint by utilizing Microsoft Graph API. The library provides easy-to-use methods for file retrieval, metadata updates, and seamless integration with SharePoint Document Libraries.

## Folder Structure

```
apgenai_sharepoint_client/
│
├── apgenai/
│   ├── sharepoint/
│   │   ├── __init__.py         # Entry point for the SharePoint library
│   │   ├── app_utils.py        # Utility functions for handling files and folders
│   │   ├── client.py           # Main SharePoint data import functionality
│   │
│   ├── tests/
│   │   ├── __init__.py         # Initializes the test package
│   │   ├── test_app_utils.py   # Tests for app_utils.py
│   │   ├── test_client.py      # Tests for client.py
│   │   ├── test_init.py        # Tests for __init__.py
│
└── README.md                   # High-level overview of the library
```

## Components

### 1. `__init__.py`
Defines the public interface of the SharePoint library by exposing key components:
- **`FileState`**: Enum for file state management (`CREATED`, `MODIFIED`, `DELETED`, `DEFAULT`).
- **`DocumentLibrary`**: Represents a SharePoint Document Library with attributes like `name`, `site_id`, `list_id`, etc.
- **`SharepointDataImport`**: The main class for managing SharePoint operations.

### 2. `app_utils.py`
Contains utility functions for environment variable handling and file/folder operations:
- **`_get_dir_details()`**: Retrieves directory paths from environment variables.
- **`_get_json_response(response)`**: Safely converts an HTTP response to JSON.
- **`_get_last_modified_subfolder(parent_folder)`**: Returns the most recently modified subfolder in a directory.

### 3. `client.py`
Implements the main functionality for interacting with SharePoint:
- **Authentication**: Uses Azure `ClientSecretCredential` for secure access.
- **Tracker Management**: Tracks files and folders retrieved or modified.
- **File Handling**:
  - **`get_all_files()`**: Recursively retrieves all files from a SharePoint Document Library.
  - **`get_files_by_modified_date()`**: Retrieves files modified after a given date.
  - **`_download_file()`**: Downloads files from SharePoint and saves them locally.
- **Metadata Handling**:
  - **`get_metadata()`**: Retrieves file metadata.
  - **`_add_row_to_metadata()`**: Adds metadata for a file or folder.
- **Summary & Logging**:
  - **`_log_summary()`**: Generates a summary of operations.
  - **`_save_df_to_parquet()`**: Saves logs or metadata as Parquet files.

## Installation

To install the library, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/apgenai_sharepoint_client.git
   cd apgenai_sharepoint_client
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Initialize the Library

```python
from apgenai.sharepoint import SharepointDataImport

# Initialize SharePoint Data Import
sp_data_import = SharepointDataImport(
    graph_site_url="https://graph.microsoft.com/v1.0/sites/{site-id}",
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant_id="your-tenant-id"
)
```

### Retrieve All Files
```python
# Get all files from the root directory
sp_data_import.get_all_files()
```

### Get Files by Modified Date
```python
# Get files modified after a specific date
sp_data_import.get_files_by_modified_date(last_modified="2024-12-01")
```

### Download a Specific File
```python
# Download a file from SharePoint
sp_data_import._download_file(
    download_url="https://graph.microsoft.com/v1.0/sites/{site-id}/drives/{drive-id}/root:/file-path:/content",
    folder_path="local_folder",
    file_name="example.txt"
)
```

### Get Metadata
```python
# Get metadata for all files
metadata = sp_data_import.get_metadata(all_data=True)
```

## Unit Tests

The library comes with unit tests to ensure code reliability and correctness. Tests are located in the `tests/` folder.

### Run Tests
To run all tests, execute:
```bash
python -m unittest discover apgenai/tests
```

## Environment Variables

The library requires the following environment variables:

| Variable Name              | Description                                |
|----------------------------|--------------------------------------------|
| `DEFAULT_LOG_DIR`          | Directory for saving logs                 |
| `DEFAULT_METADATA_DIR`     | Directory for saving metadata files       |
| `ROOT_DIR_TO_SAVE_FILES`   | Directory for saving downloaded files     |
| `SHAREPOINT_CLIENT_ID`     | Client ID for Azure authentication        |
| `SHAREPOINT_CLIENT_SECRET` | Client Secret for Azure authentication    |
| `SHAREPOINT_TENANT_ID`     | Tenant ID for Azure authentication        |

## Contributing

To contribute to the project:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes and push them:
   ```bash
   git commit -m "Add your message"
   git push origin feature/your-feature-name
   ```
4. Create a pull request.

## FAQs

1. **What is the purpose of `FileState`?**
   - It tracks the state of files (`CREATED`, `MODIFIED`, etc.) during SharePoint operations.

2. **How are logs saved?**
   - Logs are saved as Parquet files in the directory specified by `DEFAULT_LOG_DIR`.

3. **Can I use this library for sites other than SharePoint?**
   - No, this library is specifically designed for SharePoint integration via the Microsoft Graph API.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
