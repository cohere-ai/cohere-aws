import json
import time
from typing import Any, Dict, List

import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer


class ClassifyRequestSender:
    def __init__(self, endpoint_name: str):
        self._predictor = Predictor(endpoint_name, serializer=JSONSerializer())

    def send_to_endpoint(self, request):
        result = self._predictor.predict(request)
        return json.loads(result)

    def send_request(self, texts: List[str], model_id: str, verbose: bool = True):

        request: Dict[str, Any] = {"texts": texts, "adapter_id": model_id}

        t_total = time.time()
        result = self.send_to_endpoint(request)
        if verbose:
            print(f"Sending request without model weights took {time.time() - t_total:.2f}s")

        if result != "missing":
            return result

        t = time.time()
        result = self.send_to_endpoint(request)

        if verbose:
            print(f"Sending second request with model weights took {time.time() - t:.2f}s")

        return result
