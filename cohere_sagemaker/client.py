import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import tarfile
import tempfile
import sagemaker as sage
from sagemaker.s3 import parse_s3_url, S3Downloader, S3Uploader

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

from cohere_sagemaker.embeddings import Embeddings
from cohere_sagemaker.error import CohereError
from cohere_sagemaker.generation import Generation, Generations, TokenLikelihood
from cohere_sagemaker.rerank import Reranking


class Client:
    def __init__(self, endpoint_name: str = None, region_name: Optional[str] = None):
        self._endpoint_name = endpoint_name
        self._region_name = region_name
        self._client = boto3.client("sagemaker-runtime", region_name=region_name)
        self._service_client = boto3.client("sagemaker", region_name=region_name)

    def _prepare_models_dir(self, s3_models_dir) -> Tuple[str, bool]:
        if s3_models_dir.endswith(".tar.gz"):
            return s3_models_dir, False

        # If s3_models_dir is a directory, we need to tar.gz it for SageMaker
        # As this is not possible directly on s3, we download the dir to a local tmp dir, tar.gz it, and upload again
        sess = sage.Session()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download all fine-tuned models from s3
            local_models_dir = os.path.join(tmpdir, "models")
            for item in S3Downloader.list(s3_models_dir, sagemaker_session=sess):
                print(item)
                if item != s3_models_dir:
                    S3Downloader.download(item, local_models_dir, sagemaker_session=sess)
            # Tar.gz all files in downloaded dir
            model_tar = os.path.join(tmpdir, "models.tar.gz")
            with tarfile.open(model_tar, "w:gz") as tar:
                tar.add(local_models_dir, arcname=".")

            # Upload model_tar to s3
            model_tar_s3 = S3Uploader.upload(model_tar, s3_models_dir, sagemaker_session=sess)

        return model_tar_s3, True

    def connect_endpoint(self, endpoint_name: str):
        """Connects to an existing SageMaker endpoint.

        Args:
            endpoint_name (str): The name of the endpoint.

        Raises:
            CohereError: Connection to the endpoint failed.
        """
        self._endpoint_name = endpoint_name
        endpoints_response = self._service_client.list_endpoints(NameContains=self._endpoint_name)
        # Check if endpoint exists
        if len(endpoints_response["Endpoints"]) < 1:
            raise CohereError(f"Endpoint {self._endpoint_name} does not exist.")

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        s3_models_dir: str = None,
        instance_type: str = "ml.g4dn.xlarge",
        n_instances: int = 1,
        recreate: bool = False,
    ):
        """Creates and deploys a SageMaker endpoint.

        Args:
            arn (str): The product ARN. Can refer to a pre-trained model (model package) or a fine-tuned model (algorithm).
            endpoint_name (str): The name of the endpoint.
            s3_models_dir (str, optional): S3 URI pointing to fine-tuned models. Can either be an S3 folder or a .tar.gz package. Defaults to None.
            instance_type (str, optional): The EC2 instance type to deploy the endpoint to. Defaults to "ml.g4dn.xlarge".
            n_instances (int, optional): Number of endpoint instances. Defaults to 1.
            recreate (bool, optional): Force re-creation of endpoint if it already exists. Defaults to False.
        """
        self._endpoint_name = endpoint_name
        # First, check if endpoint already exists
        endpoints_response = self._service_client.list_endpoints(NameContains=self._endpoint_name)
        if len(endpoints_response["Endpoints"]) > 0:
            if recreate:
                self.delete_endpoint()
            else:
                raise CohereError(f"Endpoint {self._endpoint_name} already exists")

        kwargs = {}
        if s3_models_dir is not None:
            s3_models_dir, requires_cleanup = self._prepare_models_dir(s3_models_dir)
            # If s3_models_dir is given, we assume to have custom fine-tuned models -> Algorithm
            kwargs["algorithm_arn"] = arn
        else:
            # If no s3_models_dir is given, we assume to use a pre-trained model -> ModelPackage
            kwargs["model_package_arn"] = arn
        model = sage.ModelPackage(role="ServiceRoleSagemaker", model_data=s3_models_dir, **kwargs)
        model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)
        # If we packed & uploaded the models dir, delete it after deployment has completed
        if requires_cleanup:
            s3_resource = boto3.resource("s3")
            bucket, key = parse_s3_url(s3_models_dir)
            s3_resource.Object(bucket, key).delete()

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
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods,
            'truncate': truncate
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

    def embed(
        self,
        texts: List[str],
        truncate: str = None,
        variant: str = None
    ) -> Embeddings:
        json_params = {
            'texts': texts,
            'truncate': truncate
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

        return Embeddings(response['embeddings'])

    def rerank(self,
               query: str,
               documents: Union[List[str], List[Dict[str, Any]]],
               top_n: int = None,
                variant: str = None) -> Reranking:
        """Returns an ordered list of documents oridered by their relevance to the provided query
        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            top_n (int): (optional) The number of results to return, defaults to return all results
        """
        parsed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                parsed_docs.append({'text': doc})
            elif isinstance(doc, dict) and 'text' in doc:
                parsed_docs.append(doc)
            else:
                raise CohereError(
                    message='invalid format for documents, must be a list of strings or dicts with a "text" key')

        json_params = {
            "query": query,
            "documents": parsed_docs,
            "top_n": top_n,
            "return_documents": False
        }
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
        arn: str,
        name: str,
        train_data: str,
        s3_models_dir: str,
        eval_data: Optional[str] = None,
        # base_model: str = "english-v1",
        instance_type: str = "ml.g4dn.xlarge",
        training_parameters: Dict[str, Any] = {},  # Optional, training algorithm specific hyper-parameters
    ):
        """Creates a fine-tuning job.

        Args:
            arn (str): The product ARN of the fine-tuning package.
            name (str): The name to give to the fine-tuned model.
            train_data (str): An S3 path pointing to the training data.
            s3_models_dir (str): An S3 path pointing to the directory where the fine-tuned model will be saved.
            eval_data (str, optional): An S3 path pointing to the eval data. Defaults to None.
            instance_type (str, optional): The EC2 instance type to use for training. Defaults to "ml.g4dn.xlarge".
            training_parameters (Dict[str, Any], optional): Additional training parameters. Defaults to {}.
        """
        assert len(training_parameters) == 0  # for now we don't support any custom training parameters
        assert name != "model", "name cannot be 'model'"
        s3_models_dir = s3_models_dir + ("/" if not s3_models_dir.endswith("/") else "")

        estimator = sage.algorithm.AlgorithmEstimator(
            algorithm_arn=arn,
            role="ServiceRoleSagemaker",
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sage.Session(),
            output_path=s3_models_dir,
            hyperparameters={"name": name},
        )

        inputs = {}
        if not train_data.startswith("s3:"):
            raise ValueError("train_data must point to an S3 location.")
        inputs["training"] = train_data
        if eval_data is not None:
            if not eval_data.startswith("s3:"):
                raise ValueError("eval_data must point to an S3 location.")
            inputs["evaluation"] = eval_data
        estimator.fit(inputs=inputs)
        job_name = estimator.latest_training_job.name

        current_filepath = f"{s3_models_dir}{job_name}/output/model.tar.gz"

        s3_resource = boto3.resource("s3")

        # Copy new model to root of output_model_dir
        bucket, old_key = parse_s3_url(current_filepath)
        _, new_key = parse_s3_url(f"{s3_models_dir}{name}.tar.gz")
        s3_resource.Object(bucket, new_key).copy_from(CopySource={"Bucket": bucket, "Key": old_key})

        # Delete old dir
        bucket, old_short_key = parse_s3_url(s3_models_dir + job_name)
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
