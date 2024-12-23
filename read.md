import logging
import os
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from os.path import abspath
from os.path import basename
from os.path import dirname
from os.path import join as joinpaths
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import pandas as pd
import requests
from apgenai_utils.apgenai.utils import get_val_or_env
from app_utils import _get_dir_details, _get_json_response, _get_last_modified_subfolder
from azure.identity import ClientSecretCredential


#access the variables.
_DEFAULT_LOG_DIR, _DEFAULT_METADATA_DIR, _ROOT_DIR_TO_SAVE_FILES = _get_dir_details()

class FileState:
    CREATED = "CREATED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    DEFAULT = "DEFAULT"


class DocumentLibrary:
    def __init__(self, name, site_id, list_id, drive_id, web_url):
        self.name = name
        self.site_id = site_id
        self.list_id = list_id
        self.drive_id = drive_id
        self.web_url = web_url


class SharepointDataImport:
    def __init__(
        self,
        graph_site_url: str,
        client_id: Union[str, None] = None,
        client_secret: Union[str, None] = None,
        tenant_id: Union[str, None] = None,
        select_site_columns: Union[List[str], None] = None,
        root_dir_to_save_files: str = _ROOT_DIR_TO_SAVE_FILES,
        metadata_save_dir: str = _DEFAULT_METADATA_DIR,
        logging_dir: str = _DEFAULT_LOG_DIR,
        scope: str = "https://graph.microsoft.com/.default",
    ):
        self.graph_site_url = graph_site_url
        self.__set_credentials(client_id, client_secret, tenant_id)
        self.scope = scope
        self.__set_headers()
        self.__set_site_details()
        self.select_site_columns = select_site_columns
        self.root_dir_to_save_files = joinpaths(root_dir_to_save_files, self.site_name)
        self.metadata_save_dir = joinpaths(metadata_save_dir, self.site_name)
        self.logging_dir = joinpaths(logging_dir, self.site_name)

        self._cur_logs_saved_path = self.logging_dir
        self._cur_metadata_saved_path = self.metadata_save_dir
        self.__set_tracker()
        self.__set_metadata_df()

    def __set_credentials(
        self,
        client_id: Union[str, None] = None,
        client_secret: Union[str, None] = None,
        tenant_id: Union[str, None] = None,
    ) -> bool:
        client_id = get_val_or_env(param_val=client_id, env_key="SHAREPOINT_CLIENT_ID")
        client_secret = get_val_or_env(client_secret, "SHAREPOINT_CLIENT_SECRET")
        tenant_id = get_val_or_env(tenant_id, "SHAREPOINT_TENANT_ID")
        if not (client_id and client_secret and tenant_id):
            raise ValueError(
                "MissingParameter(s): pass all of `client_id, client_secret, tenant_id` as params to the `SharepointDataImport` constructor.\
                     \n\tOR\n Set them as environment varibales prefixed with `SHAREPOINT_`. Example `client_id` can be set as `SHAREPOINT_CLIENT_ID`."
            )
        self.credentials = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)
        return True

    def __set_tracker(self) -> bool:
        cols = [
            "TIMESTAMP_UTC","TYPE","FOLDER","NAME","STATE","URL","STATUS","MESSAGE"]
        self.__tracker = pd.DataFrame(columns=cols)
        return True

    def __set_metadata_df(self) -> bool:
        cols = ["TIMESTAMP_UTC", "TYPE", "PATH", "NAME", "STATE", "METADATA"]
        self._metadata_df = pd.DataFrame(columns=cols)
        return True

    def __add_row_to_metadata(self, row: List) -> bool:
        try:
            if not self.select_site_columns or len(self.select_site_columns) == 0:
                return False
            df_size = self._metadata_df.shape[0]
            timestamp = f"{datetime.now(timezone.utc)}"
            row = [timestamp] + row
            self._metadata_df.loc[df_size] = row
            return True
        except Exception as ex:
            logging.error(f"Failed to add row to metadata\nError: {str(ex)}")
            return False

    def get_logs(self, latest_file: bool = False, all_data: bool = False) -> pd.DataFrame:
        if latest_file:
            latest_data_path = _get_last_modified_subfolder(self.logging_dir)
            return pd.read_parquet(latest_data_path)
        if all_data:
            return pd.read_parquet(self.logging_dir)
        if self.__tracker.shape[0] > 0:
            return self.__tracker
        return pd.read_parquet(self._cur_logs_saved_path)

    def get_metadata(self, latest_file: bool = False, all_data: bool = False) -> pd.DataFrame:
        if latest_file:
            latest_data_path = _get_last_modified_subfolder(self.metadata_save_dir)
            return pd.read_parquet(latest_data_path)
        if all_data:
            return pd.read_parquet(self.metadata_save_dir)
        if self._metadata_df.shape[0] > 0:
            return self._metadata_df
        return pd.read_parquet(self._cur_metadata_saved_path)

    def __add_row_to_tracker(self, row: Union[List[str], Dict[str, str]]) -> bool:
        try:
            tracker_size = self.__tracker.shape[0]
            timestamp = f"{datetime.now(timezone.utc)}"
            row = [timestamp] + row
            self.__tracker.loc[tracker_size] = row
            return True
        except Exception as ex:
            logging.error(f"Failed to add row to tracker dataframe\nError: {str(ex)}")
            return False

    def _save_df_to_parquet(self, df: pd.DataFrame, save_dir: str) -> Union[str, None]:
        try:
            if df.shape[0] == 0:
                logging.warning("Cannot save an empty dataframe")
                return

            cur_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            path = joinpaths(save_dir, cur_timestamp, f"{cur_timestamp}.parquet")
            path = abspath(path)

            path_dir = dirname(path)
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir, exist_ok=True)
                logging.info(f"Created folder - {path_dir}")

            df.to_parquet(path=path)
            return path
        except Exception as ex:
            logging.error("Error while saving logs")
            logging.exception(ex)

    def _get_log_insights(self) -> Dict[str, int]:
        if self.__tracker.shape[0] == 0:
            logging.warning(
                f"Unable to get log insights as the current logs are empty. please check the complete logs for more info at {self.logging_dir}."
            )
            return {}
        df = self.__tracker.copy()
        insights = dict()
        insights["total_files"] = df[df["TYPE"] == "FILE"].shape[0]
        insights["imported_files_count"] = df[(df["TYPE"] == "FILE") & (df["STATUS"] == "SUCCESS")].shape[0]
        insights["folders_fails"] = df[(df["TYPE"] == "FOLDER") & (df["STATUS"] == "FAILURE")].shape[0]

        return insights

    def _log_summary(self) -> bool:
        insights = self._get_log_insights()
        if len(insights) == 0:
            return False
        total_files = insights.get("total_files", 0)
        success_files = insights.get("imported_files_count", 0)
        summary = f"Total number of files found: {total_files}\nNumber of files imported: {success_files}\n"
        if total_files != success_files:
            summary += f"Number of files failed to import: {total_files - success_files}\n"
        if insights.get("folders_fails", 0) > 0:
            summary += f"Number of folders failed to import: {insights.get('folders_fails', 0)}\n"

        logging.info(summary)
        return True

    def __set_headers(self) -> bool:
        token_info = self.credentials.get_token(self.scope)
        self.__token_expires_on = token_info.expires_on
        self.__headers = {"Authorization": "Bearer " + token_info.token}
        token_valid_till = datetime.fromtimestamp(token_info.expires_on, tz=timezone.utc)
        logging.info(f"Token is valid till: {token_valid_till} UTC")
        return True

    def __handle_token_expiry(self) -> None:
        if not self.__headers:
            logging.warn("Token info not found. Getting a new token")
            self.__set_headers()
            return None

        diff = self.__token_expires_on - time.time()
        if diff < 90:
            logging.info("Token has expired or about to expire within 90s. Getting a new token")
            self.__set_headers()
        return None

    def __get_headers(self) -> Dict[str, str]:
        self.__handle_token_expiry()
        return self.__headers

    def __set_site_details(self) -> bool:
        response = requests.get(self.graph_site_url, headers=self.__get_headers())
        # Check if the response was successful, otherwise raise error
        response.raise_for_status()
        response = _get_json_response(response)
        self.site_id = response["id"]
        self.site_name = response["name"]
        return True

    def _paginate_request(self,url: str,req_method: str = "GET",req_params: Union[Dict, None] = None,response_handler: Union[Callable, None] = None,val_key: str = "value",) -> Union[bool, List]:
        values = []
        return_values = False
        if response_handler is None:
            response_handler = lambda resp: values.extend(resp.get(val_key, []))  # noqa
            return_values = True

        # handle response, check if there are more pages and paginate
        while True:
            response = requests.request(
                method=req_method.upper(),
                url=url,
                params=req_params,
                headers=self.__get_headers(),
            )

            # Check if the response was successful, otherwise raise error
            response.raise_for_status()

            response = _get_json_response(response)
            response_handler(response)
            logging.info("Handled response from the current response")

            if response.get("@odata.nextLink"):
                url = response["@odata.nextLink"]
            else:
                break

            logging.info("Getting the next page response")

        if return_values:
            return values
        else:
            return True

    def _download_file(self,download_url: str,folder_path: str,file_name: str,state: str = FileState.DEFAULT,metadata: Union[Dict, None] = None,) -> bool:
        try:
            response = requests.get(download_url, headers=self.__get_headers())
            response.raise_for_status()
            folder_path = folder_path.strip("/\\ ")
            abs_folder_path = abspath(joinpaths(self.root_dir_to_save_files, folder_path))
            if not os.path.isdir(abs_folder_path):
                os.makedirs(abs_folder_path)
                logging.info(f"Created folder: {folder_path}")
            file_full_path = abspath(joinpaths(abs_folder_path, file_name))
            with open(file_full_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Saved/updated file at: {file_full_path}")
            self.__add_row_to_tracker(["FILE", abs_folder_path, file_name, state, download_url, "SUCCESS", ""])
            self.__add_row_to_metadata(["FILE", abs_folder_path, file_name, state, metadata])
        except Exception as ex:
            logging.error(f"Error while downloading file - {folder_path}/{file_name}")
            logging.exception(ex)
            self.__add_row_to_tracker(
                [
                    "FILE",
                    folder_path,
                    file_name,
                    download_url,
                    state,
                    "FAILURE",
                    str(ex),
                ]
            )
            return False
        return True

    def __get_all_files_response_handler(self, response: Dict) -> bool:
        """Response handler for `get_all_files` method"""
        try:
            values = response.get("value")
            for item in values:
                metadata = item.get("listItem", {}).get("fields", {})
                ch_parent_folder_path = item["parentReference"]["path"].replace("/drive/root:", "")
                local_dir = abspath(joinpaths(self.root_dir_to_save_files, ch_parent_folder_path.strip("/\\ ")))
                if "folder" in item:
                    ch_folder_path = ch_parent_folder_path.rstrip("/\\ ") + "/" + item["name"]
                    self.__add_row_to_metadata(["FOLDER", local_dir, item["name"], FileState.DEFAULT, metadata])
                    self.__get_all_files(ch_folder_path)
                elif "file" in item:
                    download_url = item["@microsoft.graph.downloadUrl"]
                    file_name = item["name"]
                    self._download_file(
                        download_url,
                        ch_parent_folder_path,
                        file_name,
                        metadata=metadata,
                    )
            return True
        except Exception as ex:
            logging.exception(ex)
            return False

    def __get_all_files(self, folder_path: str = "") -> bool:

        """Get all files from the sharepoint site recursively starting from the given folder path"""
        local_dir = abspath(joinpaths(self.root_dir_to_save_files, folder_path.strip("/\\ ")))
        try:
            request_url = ""
            root_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root"  # noqa: E231
            if folder_path == "":
                request_url = root_url + "/children"
                logging.info("Getting all files/folders from root drive")
            else:
                request_url = root_url + ":" + folder_path + ":" + "/children"
                logging.info(f"Getting all files/folders from {folder_path}")

            self.__add_row_to_tracker(
                [
                    "FOLDER",
                    dirname(local_dir),
                    basename(local_dir),
                    FileState.DEFAULT,
                    request_url,
                    "SUCCESS",
                    "",
                ]
            )
            params = None
            if self.select_site_columns and len(self.select_site_columns) > 0:
                select_params = f"listItem($expand=fields($select={','.join(self.select_site_columns)}))"
                params = {"$expand": select_params}

            self._paginate_request(
                request_url,
                req_params=params,
                response_handler=self.__get_all_files_response_handler,
            )

        except Exception as ex:
            logging.error(f"Failed to get children from the folder- {folder_path}")
            logging.exception(ex)
            self.__add_row_to_tracker(
                [
                    "FOLDER",
                    dirname(local_dir),
                    basename(local_dir),
                    request_url,
                    FileState.DEFAULT,
                    "FAILURE",
                    str(ex),
                ]
            )
            return False
        return True

    def get_all_files(self, folder_path: str = "") -> bool:
        self.__get_all_files(folder_path=folder_path)
        self._log_summary()
        self._cur_logs_saved_path = self._save_df_to_parquet(self.__tracker, save_dir=self.logging_dir)
        self.__set_tracker()
        self._cur_metadata_saved_path = self._save_df_to_parquet(self._metadata_df, save_dir=self.metadata_save_dir)
        self.__set_metadata_df()

        return self._cur_logs_saved_path

    def __handle_date_format(self, _date: Union[datetime, str, None]) -> str:
        if not _date:
            # get previous day date
            _date = datetime.now().date() - timedelta(days=1)
        if not isinstance(_date, (datetime.__base__, str)):
            raise ValueError(
                "`last_modified` parameter should be `datetime` object or a string in the format `yyyy-mm-dd`"
            )
        elif isinstance(_date, str):
            _date = datetime.strptime(_date, "%Y-%m-%d")

        _date = _date.strftime("%Y-%m-%dT%H:%M:%SZ")

        return _date

    def _get_all_lists(self, site_id=None) -> List[Dict]:
        logging.info(f"Getting all lists info from the site - {self.site_name}")
        if not site_id:
            site_id = self.site_id

        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/"  # noqa: E231

        values = self._paginate_request(url)

        return values

    def _get_all_drives(self, site_id=None) -> List[Dict]:
        logging.info(f"Getting all drives info from the site - {self.site_name}")
        if not site_id:
            site_id = self.site_id

        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/"  # noqa: E231

        values = self._paginate_request(url)

        return values

    def get_doc_lib_details(self, doc_lib_name: str):
        logging.info(f"Getting the document library - {doc_lib_name} details from the site")
        lib_details = {"name": doc_lib_name, "site_id": self.site_id}
        all_lists = self._get_all_lists(self.site_id)
        normalize_name = lambda x: x.lower().replace(" ", "")  # noqa
        doc_lib_name = normalize_name(doc_lib_name)
        for item in all_lists:
            item_name = item["webUrl"].split("/")[-1].replace("%20", " ")
            item_name = normalize_name(item_name)
            if item["list"]["template"] == "documentLibrary" and item_name == doc_lib_name:
                lib_details["list_id"] = item["id"]
                lib_details["web_url"] = item["webUrl"]
                break
        all_drives = self._get_all_drives()
        for item in all_drives:
            item_name = item["webUrl"].split("/")[-1].replace("%20", " ")
            item_name = normalize_name(item_name)
            if item_name == doc_lib_name:
                lib_details["drive_id"] = item["id"]
                if lib_details["web_url"] != item["webUrl"]:
                    logging.warning(
                        f"MismatchWebURLs: The webUrl of document library - {repr(lib_details['name'])} in lists and drives do not match.\n\
                            Lists webUrl: {lib_details['web_url']}\n\
                            Drives webUrl: {item['webUrl']}\n"
                    )
                break
        return DocumentLibrary(**lib_details)

    def __get_modified_files_response_handler(self, response: Dict):

        """Response handler for `get_files_by_modified_date` method"""
        items = response.get("value", [])
        for item in items:
            web_url = item["webUrl"].strip()
            item_type = item["contentType"]["name"].upper()
            item_path = web_url.replace(self._cur_doc_lib.web_url, "").replace("%20", " ").strip()
            item_name = item_path.split("/")[-1]
            local_dir = abspath(joinpaths(self.root_dir_to_save_files, item_path.strip("/\\ ")))
            metadata = item.get("fields", {})
            if item_type == "FOLDER":
                row = [
                    item_type,
                    local_dir,
                    item_name,
                    FileState.MODIFIED,
                    web_url,
                    "SUCCESS",
                    "",
                ]
                self.__add_row_to_tracker(row)
                self.__add_row_to_metadata(["FOLDER", local_dir, item_name, FileState.MODIFIED, metadata])
            else:
                file_name = item_path.split("/")[-1]
                folder_path = item_path.replace(file_name, "")
                download_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drives/{self._cur_doc_lib.drive_id}/root:{item_path}:/content"  # noqa: E231
                self._download_file(
                    download_url,
                    folder_path,
                    file_name,
                    state=FileState.MODIFIED,
                    metadata=metadata,
                )
        return True

    def get_files_by_modified_date(
        self,
        last_modified: Union[datetime, str, None] = None,
        doc_lib_name: str = "Shared Documents",
    ) -> bool:
        last_modified = self.__handle_date_format(last_modified)
        logging.info(f"Getting files/folders modified on/after {last_modified} from {self.site_name}")
        self._cur_doc_lib = self.get_doc_lib_details(doc_lib_name=doc_lib_name)
        url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/lists/{self._cur_doc_lib.list_id}/items"  # noqa: E231
        params = {
            "Prefer": "HonorNonIndexedQueriesWarningMayFailRandomly",
            "$filter": f"fields/Modified ge '{last_modified}'",
            "$top": 200,
        }
        if self.select_site_columns and len(self.select_site_columns) > 0:
            select_params = f"fields($select={','.join(self.select_site_columns)})"
            params["$expand"] = select_params

        self._paginate_request(
            url=url,
            req_params=params,
            response_handler=self.__get_modified_files_response_handler,
        )
        self._log_summary()
        self._cur_logs_saved_path = self._save_df_to_parquet(self.__tracker, save_dir=self.logging_dir)
        self.__set_tracker()
        self._cur_metadata_saved_path = self._save_df_to_parquet(self._metadata_df, save_dir=self.metadata_save_dir)
        self.__set_metadata_df()
        return self._cur_logs_saved_path
