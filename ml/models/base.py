import numpy as np

class BaseEstimator:
	'''
	Base class for estimators.
	'''

	def __init__(self) -> None:
		super(BaseEstimator, self).__init__()
	
	def _fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
		raise NotImplementedError()

	def _predict_proba(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError()
	
	def _predict(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError()
	
	def fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
		y = y.flatten()
		self._fit(X, y)

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		return self._predict_proba(X)

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self._predict(X)