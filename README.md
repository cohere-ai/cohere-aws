# Cohere Python SDK (AWS SageMaker)

This package provides functionality developed to simplify interfacing with the [Cohere models via AWS SageMaker Marketplace](https://aws.amazon.com/marketplace/pp/prodview-6dmzzso5vu5my) in Python >=3.6

## Installation

The package can be installed with pip:
```bash
pip install --upgrade cohere-sagemaker
```

Install from source:
```bash
python setup.py install
```

## Quick Start

To use this library, you need to configure your AWS credentials. You can do this by running `aws configure`. Once that's set up, please refer to one of the Jupyter notebooks to get started. Here is the one for our [medium command model](https://github.com/cohere-ai/cohere-sagemaker/blob/main/notebooks/Deploy%20command%20medium.ipynb)


Note: by default we assume region configured in AWS CLI (`aws configure get region`), to override use `region_name` parameter, e.g.
```python
client = Client(region_name='us-east-1')
```
