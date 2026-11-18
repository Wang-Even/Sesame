"""Mocked HTTP and metadata tests; no network calls or numerical computation."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sesame_trajectory_agent import api_client as api
from sesame_trajectory_agent.schema import Proposal


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.config = {"base_url":"https://api.deepseek.com","model":"deepseek-flash","api_key_env":"AGENT_TEST_KEY"}
        self.request = {"candidate_count":1,"robot":{"joint_names":["R1","R3","R2","R4","L1","L3","L2","L4"]},"source_clips":24}
        self.proposal = {"candidates":[Proposal().to_dict()]}
        self.envelope = {"model":"deepseek-flash","usage":{"total_tokens":12},
                         "choices":[{"finish_reason":"stop","message":{"content":json.dumps(self.proposal)}}]}

    def call(self,envelope):
        with patch.dict(os.environ,{"AGENT_TEST_KEY":"test-secret"}), patch.object(api,"_post",return_value=(envelope,200)) as post:
            result,metadata = api.call_model_with_metadata(self.request,self.config)
            payload = post.call_args.args[1]
            self.assertNotIn("test-secret",json.dumps(payload))
            self.assertEqual(payload["model"],"deepseek-flash")
            return result,metadata

    def test_valid_response(self):
        result,metadata = self.call(self.envelope)
        self.assertEqual(result,self.proposal)
        self.assertTrue(metadata["schema_valid"])
        self.assertEqual(metadata["usage"]["total_tokens"],12)

    def test_reject_blank_truncated_or_invalid_response(self):
        for content in ("   ","not json",'{"candidates":[]}',
                        '{"candidates":[{"tempo":NaN}]}','{"candidates":[{"gates":{}}]}',
                        '{"candidates":[{"tempo":0.0}]}'):
            envelope = {"choices":[{"finish_reason":"stop","message":{"content":content}}]}
            with self.subTest(content=content),self.assertRaises(api.ApiError):
                self.call(envelope)
        with self.assertRaises(api.ApiError):
            self.call({"choices":[{"finish_reason":"length","message":{"content":json.dumps(self.proposal)}}]})

    def test_http_error_does_not_disclose_response_or_secret(self):
        error = urllib.error.HTTPError("https://api.deepseek.com",401,"test-secret",{},io.BytesIO(b"test-secret"))
        with patch.object(api.urllib.request.OpenerDirector,"open",side_effect=error):
            with self.assertRaises(api.ApiError) as caught:
                api._post("https://api.deepseek.com/chat/completions",{},"test-secret")
        self.assertIn("401",str(caught.exception))
        self.assertNotIn("test-secret",str(caught.exception))

    def test_credentials_are_parsed_not_executed_and_environment_wins(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".runtime") as temp,patch.dict(os.environ,{},clear=True):
            path = Path(temp)/"test.env"
            path.write_text("AGENT_TEST_KEY=test-value\n")
            path.chmod(0o600)
            api.load_api_environment(self.config,path)
            self.assertEqual(os.environ["AGENT_TEST_KEY"],"test-value")
            os.environ["AGENT_TEST_KEY"] = "environment-value"
            api.load_api_environment(self.config,path)
            self.assertEqual(os.environ["AGENT_TEST_KEY"],"environment-value")
            del os.environ["AGENT_TEST_KEY"]
            path.chmod(0o644)
            with self.assertRaises(api.ApiError):
                api.load_api_environment(self.config,path)

    def test_endpoint_and_redirect_validation(self):
        for base in ("http://api.deepseek.com","https://user:secret@api.deepseek.com","https://api.deepseek.com?key=secret"):
            with self.subTest(base=base),self.assertRaises(api.ApiError):
                api.completion_url(self.config | {"base_url":base})
        self.assertEqual(api.completion_url(self.config),"https://api.deepseek.com/chat/completions")
        with self.assertRaises(api.ApiError):
            api._NoRedirect().redirect_request(None,None,302,"",{},"https://example.org")


if __name__ == "__main__":
    unittest.main()
