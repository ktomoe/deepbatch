"""Collection of pre-defined Metric classes."""

import numpy as np
from multiml.agent.metric import BaseMetric


class MyACCMetric(BaseMetric):
    """A metric class to return ACC."""
    def __init__(self, **kwargs):
        """Initialize ACCMetric."""
        super().__init__(**kwargs)
        self._name = 'acc'
        self._type = 'max'

    def calculate(self):
        """Calculate ACC."""
        trues = self._storegate.get_data(phase='test', var_names='labels')
        preds = self._storegate.get_data(phase='test', var_names='preds')

        preds = np.argmax(preds, axis=1)

        from sklearn.metrics import balanced_accuracy_score
        acc = balanced_accuracy_score(trues, preds)
        return acc
