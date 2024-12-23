test_app_utils.py


import unittest
from apgenai.sharepoint.app_utils import _get_dir_details, _get_json_response, _get_last_modified_subfolder
import os
import json

class TestAppUtils(unittest.TestCase):
    def test_get_dir_details(self):
        os.environ["DEFAULT_LOG_DIR"] = "/logs"
        os.environ["DEFAULT_METADATA_DIR"] = "/metadata"
        os.environ["ROOT_DIR_TO_SAVE_FILES"] = "/save_files"
        
        log_dir, metadata_dir, save_files_dir = _get_dir_details()
        self.assertEqual(log_dir, "/logs")
        self.assertEqual(metadata_dir, "/metadata")
        self.assertEqual(save_files_dir, "/save_files")

    def test_get_json_response(self):
        class MockResponse:
            def json(self):
                return {"key": "value"}
        
        response = MockResponse()
        self.assertEqual(_get_json_response(response), {"key": "value"})

    def test_get_last_modified_subfolder(self):
        os.makedirs("/tmp/test_folder", exist_ok=True)
        open("/tmp/test_folder/file.txt", "w").close()
        self.assertTrue(_get_last_modified_subfolder("/tmp/test_folder").startswith("/tmp"))

if __name__ == "__main__":
    unittest.main()


test_client.py

import unittest
from apgenai.sharepoint.client import SharepointDataImport, FileState, DocumentLibrary
from unittest.mock import patch

class TestSharepointDataImport(unittest.TestCase):
    def setUp(self):
        self.sharepoint = SharepointDataImport(
            graph_site_url="https://graph.microsoft.com/v1.0/sites/site-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id"
        )

    @patch("apgenai.sharepoint.client.ClientSecretCredential")
    def test_set_credentials(self, mock_credential):
        self.assertTrue(self.sharepoint._SharepointDataImport__set_credentials(
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id"
        ))

    def test_set_tracker(self):
        self.assertTrue(self.sharepoint._SharepointDataImport__set_tracker())

    def test_add_row_to_tracker(self):
        row = ["2024-12-23T12:00:00Z", "TYPE", "FOLDER", "NAME", "STATE", "URL", "STATUS", "MESSAGE"]
        self.assertTrue(self.sharepoint._SharepointDataImport__add_row_to_tracker(row))

if __name__ == "__main__":
    unittest.main()


test_init.py

import unittest
from apgenai.sharepoint import FileState, DocumentLibrary, SharepointDataImport

class TestInit(unittest.TestCase):
    def test_imports(self):
        self.assertIsNotNone(FileState)
        self.assertIsNotNone(DocumentLibrary)
        self.assertIsNotNone(SharepointDataImport)

if __name__ == "__main__":
    unittest.main()

