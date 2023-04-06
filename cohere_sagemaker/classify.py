import time
from pathlib import Path
from typing import Any, Dict, List, Union
import tempfile
from sagemaker.s3 import parse_s3_url

import boto3
import msgpack
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer


class ClassifyRequestSender:
    def __init__(self, endpoint_name: str):
        self._predictor = Predictor(endpoint_name, serializer=IdentitySerializer())
        self._client = boto3.client("s3")

    def send_to_endpoint(self, request):
        compressed_request = msgpack.packb(request)
        compressed_content = self._predictor.predict(compressed_request)
        uncompressed_content = msgpack.unpackb(compressed_content)
        return uncompressed_content

    def send_request(self, texts: List[str], model_id: str, model_path: Union[str, Path], verbose: bool = True):

        request: Dict[str, Any] = {"texts": texts, "adapter_id": model_id}

        t_total = time.time()
        result = self.send_to_endpoint(request)
        if verbose:
            print(f"Sending request without model weights took {time.time() - t_total:.2f}s")

        if result != "missing":
            return result

        t = time.time()
        model_path = str(model_path)
        if model_path.startswith("s3"):
            bucket, key_prefix = parse_s3_url(model_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                self._client.download_file(bucket, key_prefix, f"{tmpdir}/model.tar.gz")
                with open(f"{tmpdir}/model.tar.gz", "rb") as f:
                    request["adapter_weights"] = f.read()
        else:
            with open(model_path, "rb") as f:
                request["adapter_weights"] = f.read()
        result = self.send_to_endpoint(request)

        if verbose:
            print(f"Sending second request with model weights took {time.time() - t:.2f}s")

        return result
