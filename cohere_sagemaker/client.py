from cohere_sagemaker.classify import ClassifyRequestSender
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.s3 import parse_s3_url
import json
from typing import Any, Dict, List, Optional, Union
import sagemaker as sage

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

from cohere_sagemaker.embeddings import Embeddings
from cohere_sagemaker.error import CohereError
from cohere_sagemaker.generation import Generation, Generations, TokenLikelihood
from cohere_sagemaker.rerank import Reranking


class Client:
    def __init__(self, endpoint_name: str, region_name: Optional[str] = None, output_model_dir: Optional[str] = None):
        self._endpoint_name = endpoint_name
        self._region_name = region_name
        self._client = boto3.client("sagemaker-runtime", region_name=region_name)
        self._output_model_dir = output_model_dir  # Only used for finetuning

        if endpoint_name == "cohere-classification-finetuning":
            assert output_model_dir is not None, "output_model_dir must be specified for finetuning"

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
        stop_sequences: List[str] = None,
        return_likelihoods: str = None,
        truncate: str = None,
        variant: str = None,
    ) -> Generations:

        json_params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": k,
            "p": p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop_sequences": stop_sequences,
            "return_likelihoods": return_likelihoods,
            "truncate": truncate,
        }
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]
        json_body = json.dumps(json_params)

        params = {
            "EndpointName": self._endpoint_name,
            "ContentType": "application/json",
            "Body": json_body,
        }
        if variant is not None:
            params["TargetVariant"] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result["Body"].read().decode())
        except EndpointConnectionError as e:
            raise CohereError(e)
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(e)

        generations: List[Generation] = []
        for gen in response["generations"]:
            token_likelihoods = None

            if "token_likelihoods" in gen:
                token_likelihoods = []
                for likelihoods in gen["token_likelihoods"]:
                    token_likelihood = likelihoods["likelihood"] if "likelihood" in likelihoods else None
                    token_likelihoods.append(TokenLikelihood(likelihoods["token"], token_likelihood))
            generations.append(Generation(gen["text"], token_likelihoods))
        return Generations(generations)

    def embed(self, texts: List[str], truncate: str = None, variant: str = None) -> Embeddings:
        json_params = {"texts": texts, "truncate": truncate}
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]
        json_body = json.dumps(json_params)

        params = {
            "EndpointName": self._endpoint_name,
            "ContentType": "application/json",
            "Body": json_body,
        }
        if variant is not None:
            params["TargetVariant"] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result["Body"].read().decode())
        except EndpointConnectionError as e:
            raise CohereError(e)
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(e)

        return Embeddings(response["embeddings"])

    def rerank(
        self, query: str, documents: Union[List[str], List[Dict[str, Any]]], top_n: int = None, variant: str = None
    ) -> Reranking:
        """Returns an ordered list of documents oridered by their relevance to the provided query
        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            top_n (int): (optional) The number of results to return, defaults to return all results
        """
        parsed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                parsed_docs.append({"text": doc})
            elif isinstance(doc, dict) and "text" in doc:
                parsed_docs.append(doc)
            else:
                raise CohereError(
                    message='invalid format for documents, must be a list of strings or dicts with a "text" key'
                )

        json_params = {"query": query, "documents": parsed_docs, "top_n": top_n, "return_documents": False}
        json_body = json.dumps(json_params)

        params = {
            "EndpointName": self._endpoint_name,
            "ContentType": "application/json",
            "Body": json_body,
        }
        if variant is not None:
            params["TargetVariant"] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result["Body"].read().decode())
            reranking = Reranking(response)
            for rank in reranking.results:
                rank.document = parsed_docs[rank.index]
        except EndpointConnectionError as e:
            raise CohereError(e)
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(e)

        return reranking

    def finetune(
        self,
        name: str,
        train_data: str,
        val_data: Optional[str] = None,  # TODO: support val_data
        base_model: str = "english-v1",
        instance_type: str = "ml.g4dn.xlarge",
        training_parameters: Dict[str, Any] = {},  # Optional, training algorithm specific hyper-parameters
    ):
        if self._endpoint_name != "cohere-classification-finetuning":
            raise CohereError(f"co.finetune() is not supported for endpoint {self._endpoint_name}")

        assert base_model == "english-v1"
        assert len(training_parameters) == 0  # for now we don't support any custom training parameters
        assert name != "model", "name cannot be 'model'"

        estimator = sage.algorithm.AlgorithmEstimator(
            algorithm_arn=f"arn:aws:sagemaker:{self._region_name}:455073351313:algorithm/classification-finetuning",
            role="ServiceRoleSagemaker",
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sage.Session(),
            output_path=self._output_model_dir,
            hyperparameters={"name": name},
        )

        # TODO: train_data should work with s3:// urls too
        data_location = estimator.sagemaker_session.upload_data(train_data, key_prefix="cohere-finetune-data")
        estimator.fit({"training": data_location})
        job_name = estimator.latest_training_job.name

        output_model_dir_slash = self._output_model_dir + ("/" if not self._output_model_dir.endswith("/") else "")
        current_filepath = f"{output_model_dir_slash}{job_name}/output/model.tar.gz"

        s3_resource = boto3.resource("s3")

        # Copy new model to root of output_model_dir
        bucket, old_key = parse_s3_url(current_filepath)
        _, new_key = parse_s3_url(f"{output_model_dir_slash}{name}.tar.gz")
        s3_resource.Object(bucket, new_key).copy_from(CopySource={"Bucket": bucket, "Key": old_key})

        # Delete old dir
        bucket, old_short_key = parse_s3_url(output_model_dir_slash + job_name)
        s3_resource.Bucket(bucket).objects.filter(Prefix=old_short_key).delete()

    def deploy(self, instance_type: str = "ml.g4dn.xlarge", n_instances: int = 1):
        if self._endpoint_name == "cohere-classification-finetuning":
            estimator = sage.algorithm.AlgorithmEstimator(
                algorithm_arn=f"arn:aws:sagemaker:{self._region_name}:455073351313:algorithm/classification-finetuning",
                role="ServiceRoleSagemaker",
                instance_count=1,
                instance_type=instance_type,
                sagemaker_session=sage.Session(),
                output_path=self._output_model_dir,
            )
            estimator.deploy(initial_instance_count=n_instances, instance_type=instance_type)
        else:
            raise NotImplementedError()  # TODO

    def classify(self, input: List[str], name: str):
        if self._endpoint_name == "cohere-classification-finetuning":
            if getattr(self, "_request_sender", None) is None:
                self._request_sender = ClassifyRequestSender(self._endpoint_name, self._client)

            model_path = (
                self._output_model_dir + "/" if not self._output_model_dir.endswith("/") else "" + f"{name}.tar.gz"
            )
            return self._request_sender.send_request(input, name, model_path)
        else:
            raise ValueError(f"co.classify() is not supported for endpoint {self._endpoint_name}")

    def close(self, delete_endpoint: bool = True):
        if delete_endpoint:
            self._client.delete_endpoint(EndpointName=self._endpoint_name)
            self._client.delete_endpoint_config(EndpointConfigName=self._endpoint_name)
        self._client.close()
