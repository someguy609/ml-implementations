import math
import numpy as np
from .base import BaseEstimator
from typing import Any

class BaseNB(BaseEstimator):

	def __init__(self) -> None:
		super(BaseNB, self).__init__()
		self.priors = {}
		self.likelihoods = {}
		self.labels = []
	
	def _get_likelihood_metadata(self, X: np.ndarray) -> Any:
		raise NotImplementedError()
	
	def _get_likelihood(self, x: float, *args) -> float:
		raise NotImplementedError()
	
	def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
		self.labels = np.unique(y).tolist()
		for label in self.labels:
			label_mask = y == label
			X_y = X[label_mask, :]
			self.priors[label] = len(X_y) / len(y)
			self.likelihoods[label] = []
			for feature in X_y.T:
				metadata = self._get_likelihood_metadata(feature)
				self.likelihoods[label].append(metadata)

	def _predict_proba(self, X: np.ndarray) -> np.ndarray:
		if not self.labels:
			raise Exception()
		y_hat = np.empty((0, len(self.labels)))
		for row in X:
			label_probs = []
			for label in self.labels:
				prob = self.priors[label]
				likelihood_metadata = self.likelihoods[label]
				for feature, metadata in zip(row, likelihood_metadata):
					prob *= self._get_likelihood(feature, metadata)
				label_probs.append(prob)
			y_hat = np.vstack((y_hat, label_probs))
		return y_hat

	def _predict(self, X: np.ndarray) -> np.ndarray:
		probas = self._predict_proba(X)
		return np.argmax(probas, axis=1)
			

class GaussianNB(BaseNB):

	def __init__(self, *args, **kwargs) -> None:
		super(GaussianNB, self).__init__(*args, **kwargs)
	
	def _get_likelihood_metadata(self, X: np.ndarray) -> dict[str, float]:
		return {
			'mean': X.mean(),
			'std': X.std()
		}

	def _get_likelihood(self, x: float, metadata: dict[str, float]) -> float:
		mean = metadata['mean']
		std = metadata['std']
		if std == 0:
			return 1 if x == mean else 1e-9
		Z = (x - mean) / std
		p = 1 / (math.sqrt(2 * math.pi) * std) * np.exp(-0.5 * Z ** 2)
		return p
