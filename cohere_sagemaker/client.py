import json
import os
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import sagemaker as sage
from botocore.exceptions import (ClientError, EndpointConnectionError,
                                 ParamValidationError)
from sagemaker.s3 import S3Downloader, S3Uploader, parse_s3_url

from cohere_sagemaker.classification import Classification, Classifications
from cohere_sagemaker.embeddings import Embeddings
from cohere_sagemaker.error import CohereError
from cohere_sagemaker.generation import (Generation, Generations,
                                         TokenLikelihood)
from cohere_sagemaker.rerank import Reranking
from cohere_sagemaker.summary import Summary


class Client:
    def __init__(self, endpoint_name: Optional[str] = None, region_name: Optional[str] = None):
        """
        By default we assume region configured in AWS CLI (`aws configure get region`). You can change the region with
        `aws configure set region us-west-2` or override it with `region_name` parameter.
        """
        self._endpoint_name = endpoint_name  # deprecated, should use self.connect_to_endpoint() instead
        self._client = boto3.client("sagemaker-runtime", region_name=region_name)
        self._service_client = boto3.client("sagemaker", region_name=region_name)
        self._sess = sage.Session(sagemaker_client=self._service_client)

    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        try:
            self._service_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            return False
        return True

    def connect_to_endpoint(self, endpoint_name: str) -> None:
        """Connects to an existing SageMaker endpoint.

        Args:
            endpoint_name (str): The name of the endpoint.

        Raises:
            CohereError: Connection to the endpoint failed.
        """
        if not self._does_endpoint_exist(endpoint_name):
            raise CohereError(f"Endpoint {endpoint_name} does not exist.")
        self._endpoint_name = endpoint_name

    def _s3_models_dir_to_tarfile(self, s3_models_dir: str) -> str:
        """
        Compress an S3 folder which contains one or several fine-tuned models to a tar file.
        If the S3 folder contains only one fine-tuned model, it simply returns the path to that model.
        If the S3 folder contains several fine-tuned models, it download all models, aggregates them into a single
        tar.gz file.

        Args:
            s3_models_dir (str): S3 URI pointing to a folder

        Returns:
            str: S3 URI pointing to the `models.tar.gz` file
        """

        s3_models_dir = s3_models_dir + ("/" if not s3_models_dir.endswith("/") else "")

        # Links of all fine-tuned models in s3_models_dir. Their format should be .tar.gz
        s3_tar_models = [
            s3_path
            for s3_path in S3Downloader.list(s3_models_dir, sagemaker_session=self._sess)
            if (
                s3_path.endswith(".tar.gz")  # only .tar.gz files
                and (s3_path.split("/")[-1] != "models.tar.gz")  # exclude the .tar.gz file we are creating
                and (s3_path.rsplit("/", 1)[0] == s3_models_dir[:-1])  # only files at the root of s3_models_dir
            )
        ]

        if len(s3_tar_models) == 0:
            raise CohereError(f"No fine-tuned models found in {s3_models_dir}")
        elif len(s3_tar_models) == 1:
            print(f"Found one fine-tuned model: {s3_tar_models[0]}")
            return s3_tar_models[0]

        # More than one fine-tuned model found, need to aggregate them into a single .tar.gz file
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tar_models_dir = os.path.join(tmpdir, "tar")
            local_models_dir = os.path.join(tmpdir, "models")

            # Download and extract all fine-tuned models
            for s3_tar_model in s3_tar_models:
                print(f"Adding fine-tuned model: {s3_tar_model}")
                S3Downloader.download(s3_tar_model, local_tar_models_dir, sagemaker_session=self._sess)
                with tarfile.open(os.path.join(local_tar_models_dir, s3_tar_model.split("/")[-1])) as tar:
                    tar.extractall(local_models_dir)

            # Compress local_models_dir to a tar.gz file
            model_tar = os.path.join(tmpdir, "models.tar.gz")
            with tarfile.open(model_tar, "w:gz") as tar:
                tar.add(local_models_dir, arcname=".")

            # Upload the new tarfile containing all models to s3
            # Very important to remove the trailing slash from s3_models_dir otherwise it just doesn't upload
            model_tar_s3 = S3Uploader.upload(model_tar, s3_models_dir[:-1], sagemaker_session=self._sess)

            # sanity check
            assert s3_models_dir + "models.tar.gz" in S3Downloader.list(s3_models_dir, sagemaker_session=self._sess)

        return model_tar_s3

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        s3_models_dir: Optional[str] = None,
        instance_type: str = "ml.g4dn.xlarge",
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
    ) -> None:
        """Creates and deploys a SageMaker endpoint.

        Args:
            arn (str): The product ARN. Refers to a ready-to-use model (model package) or a fine-tuned model
                (algorithm).
            endpoint_name (str): The name of the endpoint.
            s3_models_dir (str, optional): S3 URI pointing to the folder containing fine-tuned models. Defaults to None.
            instance_type (str, optional): The EC2 instance type to deploy the endpoint to. Defaults to "ml.g4dn.xlarge".
            n_instances (int, optional): Number of endpoint instances. Defaults to 1.
            recreate (bool, optional): Force re-creation of endpoint if it already exists. Defaults to False.
            rool (str, optional): The IAM role to use for the endpoint. If not provided, sagemaker.get_execution_role()
                will be used to get the role. This should work when one uses the client inside SageMaker. If this errors
                out, the default role "ServiceRoleSagemaker" will be used, which generally works outside of SageMaker.
        """
        # First, check if endpoint already exists
        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name)
                self.delete_endpoint()
            else:
                raise CohereError(f"Endpoint {endpoint_name} already exists and recreate={recreate}.")

        kwargs = {}
        model_data = None
        if s3_models_dir is not None:
            # If s3_models_dir is given, we assume to have custom fine-tuned models -> Algorithm
            kwargs["algorithm_arn"] = arn
            model_data = self._s3_models_dir_to_tarfile(s3_models_dir)
        else:
            # If no s3_models_dir is given, we assume to use a pre-trained model -> ModelPackage
            kwargs["model_package_arn"] = arn

        # Out of precaution, check if there is an endpoint config and delete it if that's the case
        # Otherwise it might block deployment
        try:
            self._service_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError:
            pass

        if role is None:
            try:
                role = sage.get_execution_role()
            except ValueError:
                print("Using default role: 'ServiceRoleSagemaker'.")
                role = "ServiceRoleSagemaker"

        model = sage.ModelPackage(
            role=role,
            model_data=model_data,
            sagemaker_session=self._sess,  # makes sure the right region is used
            **kwargs
        )

        validation_params = dict(
            model_data_download_timeout=2400,
            container_startup_health_check_timeout=2400
        )

        try:
            model.deploy(
                n_instances,
                instance_type,
                endpoint_name=endpoint_name,
                **validation_params
            )
        except ParamValidationError:
            # For at least some versions of python 3.6, SageMaker SDK does not support the validation_params
            model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)
        self.connect_to_endpoint(endpoint_name)

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
        stop_sequences: Optional[List[str]] = None,
        return_likelihoods: Optional[str] = None,
        truncate: Optional[str] = None,
        variant: Optional[str] = None
    ) -> Generations:

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        json_params = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
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
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

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
        truncate: Optional[str] = None,
        variant: Optional[str] = None
    ) -> Embeddings:

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

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
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return Embeddings(response['embeddings'])

    def rerank(self,
               query: str,
               documents: Union[List[str], List[Dict[str, Any]]],
               top_n: Optional[int] = None,
               variant: Optional[str] = None,
               max_chunks_per_doc: Optional[int] = None) -> Reranking:
        """Returns an ordered list of documents oridered by their relevance to the provided query
        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            top_n (int): (optional) The number of results to return, defaults to return all results
            max_chunks_per_doc (int): (optional) The maximum number of chunks derived from a document
        """

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

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
            "return_documents": False,
            "max_chunks_per_doc" : max_chunks_per_doc
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
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return reranking

    def classify(self, input: List[str], name: str) -> Classifications:

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        json_params = {"texts": input, "model_id": name}
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
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return Classifications([Classification(classification) for classification in response])

    def create_finetune(
        self,
        arn: str,
        name: str,
        train_data: str,
        s3_models_dir: str,
        eval_data: Optional[str] = None,
        instance_type: str = "ml.g4dn.xlarge",
        training_parameters: Dict[str, Any] = {},  # Optional, training algorithm specific hyper-parameters
        role: Optional[str] = None,
    ) -> None:
        """Creates a fine-tuning job.

        Args:
            arn (str): The product ARN of the fine-tuning package.
            name (str): The name to give to the fine-tuned model.
            train_data (str): An S3 path pointing to the training data.
            s3_models_dir (str): An S3 path pointing to the directory where the fine-tuned model will be saved.
            eval_data (str, optional): An S3 path pointing to the eval data. Defaults to None.
            instance_type (str, optional): The EC2 instance type to use for training. Defaults to "ml.g4dn.xlarge".
            training_parameters (Dict[str, Any], optional): Additional training parameters. Defaults to {}.
            rool (str, optional): The IAM role to use for the endpoint. If not provided, sagemaker.get_execution_role()
                will be used to get the role. This should work when one uses the client inside SageMaker. If this errors
                out, the default role "ServiceRoleSagemaker" will be used, which generally works outside of SageMaker.
        """
        assert name != "model", "name cannot be 'model'"
        s3_models_dir = s3_models_dir + ("/" if not s3_models_dir.endswith("/") else "")

        if role is None:
            try:
                role = sage.get_execution_role()
            except ValueError:
                print("Using default role: 'ServiceRoleSagemaker'.")
                role = "ServiceRoleSagemaker"

        training_parameters.update({"name": name})
        estimator = sage.algorithm.AlgorithmEstimator(
            algorithm_arn=arn,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=self._sess,
            output_path=s3_models_dir,
            hyperparameters=training_parameters,
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
        s3_resource.Object(bucket, new_key).copy(CopySource={"Bucket": bucket, "Key": old_key})

        # Delete old dir
        bucket, old_short_key = parse_s3_url(s3_models_dir + job_name)
        s3_resource.Bucket(bucket).objects.filter(Prefix=old_short_key).delete()

    def summarize(
        self,
        text: str,
        length: Optional[str] = "auto",
        format_: Optional[str] = "auto",
        # Only summarize-xlarge is supported on Sagemaker
        # model: Optional[str] = "summarize-xlarge",
        extractiveness: Optional[str] = "auto",
        temperature: Optional[float] = 0.3,
        additional_command: Optional[str] = "",
        variant: Optional[str] = None
    ) -> Summary:

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        json_params = {
            'text': text,
            'length': length,
            'format': format_,
            'extractiveness': extractiveness,
            'temperature': temperature,
            'additional_command': additional_command,
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
            summary = Summary(response)
        except EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return summary


    def delete_endpoint(self) -> None:
        if self._endpoint_name is None:
            raise CohereError("No endpoint connected.")
        try:
            self._service_client.delete_endpoint(EndpointName=self._endpoint_name)
        except:
            print("Endpoint not found, skipping deletion.")

        try:
            self._service_client.delete_endpoint_config(EndpointConfigName=self._endpoint_name)
        except:
            print("Endpoint config not found, skipping deletion.")

    def close(self) -> None:
        try:
            self._client.close()
            self._service_client.close()
        except AttributeError:
            print("SageMaker client could not be closed. This might be because you are using an old version of SageMaker.")
            raise
