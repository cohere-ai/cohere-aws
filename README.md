# Cohere Python SDK (AWS SageMaker)

This package provides functionality developed to simplify interfacing with the [Cohere models via AWS SageMaker Marketplace](https://aws.amazon.com/marketplace/pp/prodview-6dmzzso5vu5my) in Python 3.

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

To use this library, you need to configure your AWS credentials. You can do this by running `aws configure`. Once that's set up, you can use the library like this:
```python
from cohere_sagemaker import Client

client = Client(endpoint_name='my-cohere-endpoint')

# generate prediction for a prompt
response = client.generate(
    prompt="Tell me a story about",
    max_tokens=20)

print(response.generations[0].text)
# a time when you had to make a difficult decision. What did you do
```
Note: by default we assume region configured in AWS CLI (`aws configure get region`), to override use `region_name` parameter, e.g.
```python
client = Client(endpoint_name='my-cohere-endpoint', region_name='us-east-1')
```

More detailed examples can be found in the [Jupyter notebook](https://github.com/cohere-ai/cohere-sagemaker/blob/main/notebooks/Deploy%20cohere%20model.ipynb).