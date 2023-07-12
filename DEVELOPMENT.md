
# Publish to PyPi

Bump version in `setup.py` then:
```bash
python setup.py sdist
python -m twine upload dist/*
```
If you have 2FA enabled (which you should), follow [these instructions](https://pypi.org/help/#apitoken) for the `twine upload` username and password.

Check releases on [PyPi](https://pypi.org/project/cohere-sagemaker/#history).
