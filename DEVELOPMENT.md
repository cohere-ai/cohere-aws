
# Publish to PyPi

Bump version in `setup.py` then:
```bash
python setup.py sdist
python -m twine upload dist/*
```
Check releases on [PyPi](https://pypi.org/project/cohere-sagemaker/#history).
