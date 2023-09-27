import unittest
import json

from cohere_aws import Client, CohereError, Mode
from botocore.stub import Stubber
from botocore.response import StreamingBody
from io import BytesIO
from typing import Dict, Optional, Any


class TestBedrockClient(unittest.TestCase):
    MODEL_ID= "model-id"
    PROMPT = "Hello world!"
    EXPECTED_ACTUAL_GENERATION = [
        " Hello", "!",
        " How", " can", " I", " help", " you", " today", "?", "<EOS_TOKEN>"]
    TEXT = "Mock generation #1"
    TEXT2 = "Mock generation #2"

    def setUp(self):
        self.client = Client(mode=Mode.BEDROCK)
        self.default_request_params = {"prompt": self.PROMPT,
                                       "max_tokens": 20,
                                       "temperature": 1.0,
                                       "k": 0,
                                       "p": 0.75,
                                       #"num_generations": 1,
                                       "stop_sequences": None,
                                       "return_likelihoods": None,
                                       "truncate": None,
                                       "stream": False}
        super().setUp()

    def tearDown(self):
        self.client.close()

    def stub(self, expected_params, generations_text):
        stubber = Stubber(self.client._client)
        generations = []
        for text in generations_text:
            generations.append(f'{{"text": "{text}"}}')
        generations = ', '.join(generations)
        b = f'{{"generations": [{generations}]}}'.encode()
        mock_response = {"body": StreamingBody(BytesIO(b), len(b)), "contentType": "application/json"}
        stubber.add_response('invoke_model', mock_response, expected_params)
        stubber.activate()

    def stub_err(self, service_message, http_status_code=400):
        stubber = Stubber(self.client._client)
        stubber.add_client_error('invoke_model_with_response_stream',
                                 service_message=service_message,
                                 http_status_code=http_status_code)
        stubber.activate()

    def expected_params(self,
                        custom_request_params: Optional[Dict[str, Any]] = {}
                        ) -> Dict:
        # Optionally override the default parameters with custom ones
        for k, v in custom_request_params.items():
            self.default_request_params[k] = v
        # Remove null parameters
        self.default_request_params = {
            k: v for k, v in self.default_request_params.items()
            if v is not None}

        return {"body": f'{json.dumps(self.default_request_params)}',
                "modelId": self.MODEL_ID}

    def test_generate_defaults(self):
        self.stub(self.expected_params(), [self.TEXT])
        response = self.client.generate(
            self.PROMPT, stream=False, model_id=self.MODEL_ID)
        self.assertEqual(len(response.generations), 1)
        self.assertEqual(response.generations[0]['text'], self.TEXT)

    def test_streaming(self):
        stream = self.client.generate(
            self.PROMPT, stream=True, temperature=0, max_tokens=10,
            model_id="cohere.command-text-v14")
        for i, s in enumerate(stream):
            self.assertEqual(s.text, self.EXPECTED_ACTUAL_GENERATION[i])

    def test_override_defaults(self):
        self.stub(self.expected_params(
            custom_request_params={"temperature": 0.0,
                                   "max_tokens": 40,
                                   "k": 1,
                                   "p": 0.5,
                                   "stop_sequences": ["."],
                                   "return_likelihoods": "likelihood",
                                   "truncate": "LEFT"}),
                  [self.TEXT])
        response = self.client.generate(self.PROMPT,
                                        temperature=0.0,
                                        max_tokens=40,
                                        k=1,
                                        p=0.5,
                                        stop_sequences=["."],
                                        return_likelihoods="likelihood",
                                        truncate="LEFT",
                                        stream=False,
                                        model_id=self.MODEL_ID)
        self.assertEqual(len(response.generations), 1)
        self.assertEqual(response.generations[0]['text'], self.TEXT)

    def test_missing_model_id(self):
        expected_err = "must supply model_id"
        self.stub_err(expected_err)
        try:
            self.client.generate(self.PROMPT, model_id=self.MODEL_ID)
            self.fail("expected error")
        except CohereError as e:
            self.assertIn(expected_err,
                          str(e.message))

    def test_bad_region(self):
        expected_err = "Could not connect to the endpoint URL"
        self.stub_err(expected_err)
        try:
            self.client.generate(self.PROMPT, model_id=self.MODEL_ID)
            self.fail("expected error")
        except CohereError as e:
            self.assertIn(expected_err,
                          str(e.message))

if __name__ == '__main__':
    unittest.main()
