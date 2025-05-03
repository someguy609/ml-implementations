import numpy as np
from .base import BaseEstimator
from typing import Optional


class LogisticRegression(BaseEstimator):
	'''
	Logistic Regression.
	
	Args:
		intercept (bool, optional): Enable intercept. Default value is False.
		learning_rate (float, optional): Training learning rate. Default value is 0.01.
		max_iter (int, optional): Maximum iterations for gradient descent. Default value is 100.
		threshold (bool, optional): Proba threshold to be considered True. Default value is 0.5.
		random_seed (int, optional): Random seed for parameter initialization. Default value is None.
		verbose (bool, optional): Enable verbose output. Default value is False.
	'''

	def __init__(
		self,
		intercept: Optional[bool] = True,
		learning_rate: Optional[float] = 1e-2,
		max_iter: Optional[int] = 100,
		threshold: Optional[float] = 0.5,
		random_seed: Optional[int] = None,
		verbose: Optional[bool] = False
	) -> None:
		super(LogisticRegression, self).__init__()
		self.intercept = intercept
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.threshold = threshold
		self.random_seed = random_seed
		self.verbose = verbose

		self.w = None

	def _fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
	) -> None:
		'''
		Args:
			X (np.ndarray): The training set features.
			y (np.ndarray): The training set targets.
		'''

		np.random.seed(self.random_seed)
		num_samples, num_features = X.shape
		self.w = np.random.randn(num_features + self.intercept)

		for i in range(self.max_iter):
			y_hat = self._predict_proba(X)[:, 1]
			grad = np.dot(X.T, y_hat - y) / num_samples
			self.w -= self.learning_rate * grad
			if self.verbose:
				error = -(y * np.log(y_hat) + (1 - y)
						  * np.log(1 - y_hat)).mean()
				print(f'Step {i + 1}, Error: {error:.4f}')

	def _predict_proba(self, X: np.ndarray) -> np.ndarray:
		'''
		Args:
			X (np.ndarray): The set of features.
		'''
		
		if self.w is None:
			raise Exception()
		if self.intercept:
			ones = np.ones((X.shape[0], 1))
			X = np.hstack([ones, X])
		prob = 1 / (1 + np.exp(-np.dot(X, self.w)))
		prob = prob.reshape(X.shape[0], 1)
		return np.hstack([1 - prob, prob])

	def _predict(self, X: np.ndarray) -> np.ndarray:
		'''
		Args:
			X (np.ndarray): The set of features.
		'''
		
		return (self._predict_proba(X) >= self.threshold).astype(int).astype(int)