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
        prompt: str = None,
        model: str = None,
        preset: str = None,
        num_generations: int = 1,
        max_tokens: int = 20,
        temperature: float = 1.0,
        k: int = 0,
        p: float = 0.75,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: List[str] = None,
        return_likelihoods: str = None, # this changed from 'NONE'
        truncate: str = None,
        variant: str = None
    ) -> Generations:
        json_body = json.dumps({
            'model': model,
            'prompt': prompt,
            'preset': preset,
            'num_generations': num_generations,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods,
            'truncate': truncate,
        })

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
            likelihood = None
            token_likelihoods = None
            if return_likelihoods == 'GENERATION' or return_likelihoods == 'ALL':
                likelihood = gen['likelihood']
            if 'token_likelihoods' in gen.keys():
                token_likelihoods = []
                for likelihoods in gen['token_likelihoods']:
                    token_likelihood = likelihoods['likelihood'] if 'likelihood' in likelihoods.keys() else None
                    token_likelihoods.append(TokenLikelihood(likelihoods['token'], token_likelihood))
            generations.append(Generation(gen['text'], likelihood, token_likelihoods))
        return Generations(generations, return_likelihoods)

    def close(self):
        self._client.close()
