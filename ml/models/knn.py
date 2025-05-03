import numpy as np
from .base import BaseEstimator
from scipy.spatial import distance
from scipy.stats import mode
from typing import Optional

class BaseKNN(BaseEstimator):

	def __init__(
		self,
		num_neighbors: Optional[int] = 5,
		p: Optional[int] = 2
	) -> None:
		super(BaseKNN, self).__init__()
		self.num_neighbors = num_neighbors
		self.p = p
	
	def _fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		self.data = X
		self.targets = y

	def _predict(self, X: np.ndarray) -> np.ndarray:
		if self.data is None:
			raise Exception()
		distances = distance.cdist(X, self.data, metric='minkowski', p=self.p)
		i = np.argpartition(distances, self.num_neighbors, axis=1)[:, :self.num_neighbors]
		neighbors = self.targets[i]
		return neighbors

class KNNClassifier(BaseKNN):

	def __init__(self, *args, **kwargs) -> None:
		super(KNNClassifier, self).__init__(*args, **kwargs)
	
	def _predict(self, X: np.ndarray) -> np.ndarray:
		neighbors = super(KNNClassifier, self)._predict(X)
		y_hat = mode(neighbors, axis=1, keepdims=True)[0].ravel()
		return y_hat

class KNNRegressor(BaseKNN):

	def __init__(self, *args, **kwargs) -> None:
		super(KNNRegressor, self).__init__(*args, **kwargs)
	
	def _predict(self, X) -> np.ndarray:
		neighbors = super(KNNRegressor, self)._predict(X)
		y_hat = neighbors.mean(axis=1)
		return y_hat