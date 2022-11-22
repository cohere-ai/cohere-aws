# Cohere Python SDK (AWS SageMaker)

## Install

```
pip install cohere-sagemaker
```

## Usage

```python
from cohere_sagemaker import Client

client = Client(endpoint_name='my-cohere-endpoint')
response = client.generate(prompt="Tell me a story about")

print(response.generations[0].text)
# a time when you had to make a difficult decision. What did you do
```
