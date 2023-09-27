# Cohere Python SDK (AWS SageMaker & Bedrock)

This package provides functionality developed to simplify interfacing with the [Cohere models via AWS SageMaker Marketplace](https://aws.amazon.com/marketplace/pp/prodview-6dmzzso5vu5my) and the [Cohere models via AWS Bedrock](https://aws.amazon.com/marketplace/pp/prodview-r6zvppobprqmy) in Python >=3.9

## Installation

The package can be installed with pip:
```bash
pip install --upgrade cohere-aws
```

Install from source:
```bash
python setup.py install
```

## Quick Start

To use this library, you need to configure your AWS credentials. You can do this by running `aws configure`. Once that's set up, please refer to one of the Jupyter notebooks to get started. Here is the one for our [medium command model](https://github.com/cohere-ai/cohere-aws/blob/main/notebooks/sagemaker/Deploy%20command%20medium.ipynb)


Note: by default we assume region configured in AWS CLI (`aws configure get region`), to override use `region_name` parameter, e.g.
```python
client = Client(region_name='us-east-1')
```
