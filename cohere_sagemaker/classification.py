from cohere_sagemaker.response import CohereObject
from typing import Iterator, List, Union


class Classification(CohereObject):
    def __init__(self, classification: Union[str, int, List[str], List[int]]) -> None:
        # A classification can be either a label (int or string) for single-label classification,
        # or a list of labels (int or string) for multi-label classification.
        self.classification = classification

    def is_multilabel(self) -> bool:
        return not isinstance(self.classification, (int, str))


class Classifications(CohereObject):
    def __init__(self, classifications: List[Classification]) -> None:
        self.classifications = classifications
        if len(self.classifications) > 0:
            assert all(
                [c.is_multilabel() == self.is_multilabel() for c in self.classifications]
            ), "All classifications must be of the same type (single-label or multi-label)"

    def __iter__(self) -> Iterator:
        return iter(self.classifications)

    def __len__(self) -> int:
        return len(self.classifications)

    def is_multilabel(self) -> bool:
        return len(self.classifications) > 0 and self.classifications[0].is_multilabel()
