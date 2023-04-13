import json
import os
from typing import Any, Dict, List, Optional, Union
import tarfile
import tempfile
import sagemaker as sage
from sagemaker.s3 import parse_s3_url, S3Downloader

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

from cohere_sagemaker.embeddings import Embeddings
from cohere_sagemaker.error import CohereError
from cohere_sagemaker.generation import Generation, Generations, TokenLikelihood
from cohere_sagemaker.rerank import Reranking


class Client:
    def __init__(
        self,
        endpoint_name: str = None,
        region_name: Optional[str] = None,
    ):
        self._endpoint_name = endpoint_name
        self._region_name = region_name
        self._client = boto3.client("sagemaker-runtime", region_name=region_name)
        self._service_client = boto3.client("sagemaker", region_name=region_name)
        self._s3_upload_prefix = "cohere-finetune-data"

    def _ensure_output_model_dir_created(self):
        s3 = boto3.resource("s3")
        bucket, key_prefix = parse_s3_url(self._output_model_dir)
        s3.Bucket(bucket).put_object(Key=f"{key_prefix}")

    def _prepare_data(self, sess, data_path, prefix):
        if data_path.startswith("s3"):
            return data_path
        else:
            return sess.upload_data(data_path, key_prefix=self._s3_upload_prefix + "/" + prefix)

    def _prepare_models_dir(self, models_dir):
        if models_dir.endswith(".tar.gz"):
            return models_dir

        sess = sage.Session()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download all fine-tuned models from s3
            local_models_dir = os.path.join(tmpdir, "models")
            for item in S3Downloader.list(models_dir, sagemaker_session=sess):
                print(item)
                if item != models_dir:
                    S3Downloader.download(item, local_models_dir, sagemaker_session=sess)
            # Tar.gz all files in downloaded dir
            model_tar = os.path.join(tmpdir, "models.tar.gz")
            with tarfile.open(model_tar, "w:gz") as tar:
                tar.add(local_models_dir, arcname=".")

            # Upload model_tar to s3
            model_tar_s3 = sess.upload_data(model_tar, key_prefix=self._s3_upload_prefix)

        return model_tar_s3

    def connect_endpoint(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        # TODO maybe check if endpoint exists?

    def create_endpoint(
        self,
        model_package_arn: str,
        endpoint_name: str,
        models_dir: str = None,
        instance_type: str = "ml.g4dn.xlarge",
        n_instances: int = 1,
        recreate: bool = False,
    ):
        self._endpoint_name = endpoint_name
        endpoints_response = self._service_client.list_endpoints(NameContains=self._endpoint_name)
        if len(endpoints_response["Endpoints"]) > 0:
            if recreate:
                self.delete_endpoint()
            else:
                raise CohereError(f"Endpoint {self._endpoint_name} already exists")

        models_dir = self._prepare_models_dir(models_dir)
        model = sage.ModelPackage(
            role="ServiceRoleSagemaker",
            model_data=models_dir,  # Required arg, may point to an empty dir
            algorithm_arn=model_package_arn,
        )
        model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)

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

    def create_finetune(
        self,
        model_package_arn: str,
        name: str,
        train_data: str,
        models_dir: str,
        eval_data: Optional[str] = None,
        # base_model: str = "english-v1",
        instance_type: str = "ml.g4dn.xlarge",
        training_parameters: Dict[str, Any] = {},  # Optional, training algorithm specific hyper-parameters
    ):
        assert len(training_parameters) == 0  # for now we don't support any custom training parameters
        assert name != "model", "name cannot be 'model'"
        models_dir = models_dir + ("/" if not models_dir.endswith("/") else "")

        estimator = sage.algorithm.AlgorithmEstimator(
            algorithm_arn=model_package_arn,
            role="ServiceRoleSagemaker",
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sage.Session(),
            output_path=models_dir,
            hyperparameters={"name": name},
        )

        inputs = {}
        inputs["training"] = self._prepare_data(estimator.sagemaker_session, train_data, "training")
        if eval_data is not None:
            inputs["evaluation"] = self._prepare_data(estimator.sagemaker_session, eval_data, "validation")
        estimator.fit(inputs=inputs)
        job_name = estimator.latest_training_job.name

        current_filepath = f"{models_dir}{job_name}/output/model.tar.gz"

        s3_resource = boto3.resource("s3")

        # Copy new model to root of output_model_dir
        bucket, old_key = parse_s3_url(current_filepath)
        _, new_key = parse_s3_url(f"{models_dir}{name}.tar.gz")
        s3_resource.Object(bucket, new_key).copy_from(CopySource={"Bucket": bucket, "Key": old_key})

        # Delete old dir
        bucket, old_short_key = parse_s3_url(models_dir + job_name)
        s3_resource.Bucket(bucket).objects.filter(Prefix=old_short_key).delete()

    def classify(self, input: List[str], name: str):
        json_params = {"texts": input, "adapter_id": name}
        json_body = json.dumps(json_params)

        params = {
            "EndpointName": self._endpoint_name,
            "ContentType": "application/json",
            "Body": json_body,
        }

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result["Body"].read().decode())
        except EndpointConnectionError as e:
            raise CohereError(e)
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(e)

        return response

    def delete_endpoint(self):
        self._service_client.delete_endpoint(EndpointName=self._endpoint_name)
        self._service_client.delete_endpoint_config(EndpointConfigName=self._endpoint_name)

    def close(self):
        self._client.close()
        self._service_client.close()
