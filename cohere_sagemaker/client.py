import json
from typing import List

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

from cohere_sagemaker.generation import Generations, Generation, TokenLikelihood
from cohere_sagemaker.error import CohereError

class Client:

    def __init__(self, endpoint_name: str, region_name: str = None):
        self._endpoint_name = endpoint_name
        self._client = boto3.client(
            'sagemaker-runtime',
            region_name=region_name)

    def generate(
        self,
        prompt: str,
        # not applicable to sagemaker deployment
        # model: str = None,
        # requires DB with presets
        # preset: str = None,
        # not implemented in API
        # num_generations: int = 1,
        max_tokens: int = 20,
        temperature: float = 1.0,
        k: int = 0,
        p: float = 0.75,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        # not implemented in API
        # stop_sequences: List[str] = None,
        # not implemented in API
        # return_likelihoods: str = None,
        # not implemented in API
        # truncate: str = None,
        variant: str = None
    ) -> Generations:

        json_params = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
        }
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]
        json_body = json.dumps(json_params)

        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant is not None:
            params['TargetVariant'] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result['Body'].read().decode())
        except EndpointConnectionError as e:
            raise CohereError(e)
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad 
            raise CohereError(e)

        generations: List[Generation] = []
        for gen in response['generations']:
            token_likelihoods = None
                
            if 'token_likelihoods' in gen:
                token_likelihoods = []
                for likelihoods in gen['token_likelihoods']:
                    token_likelihood = likelihoods['likelihood'] if 'likelihood' in likelihoods else None
                    token_likelihoods.append(TokenLikelihood(likelihoods['token'], token_likelihood))
            generations.append(Generation(gen['text'], token_likelihoods))
        return Generations(generations)

    def close(self):
        self._client.close()
