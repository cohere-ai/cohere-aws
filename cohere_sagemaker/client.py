import json

import boto3
from botocore.exceptions import ClientError

class Client:

    def __init__(self, endpoint_name: str, region_name: str = None):
        self._endpoint_name = endpoint_name
        self._client = boto3.client(
            'sagemaker-runtime',
            region_name=region_name)
    
    def generate(self, prompt: str, variant: str = None):
        json_body = json.dumps({
            'prompt': prompt,
            })

        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant is not None:
            params['TargetVariant'] = variant

        try:
            return self._client.invoke_endpoint(**params)
        except ClientError as e:
            print(f"Error calling Cohere endpoint: {e}")
