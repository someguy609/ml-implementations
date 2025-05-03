import numpy as np
from .base import BaseEstimator
from typing import Literal, Optional


class LinearRegression(BaseEstimator):
	'''
	Linear regression.

	Args:
		intercept (bool, optional): Enable intercept. Default value is True.
		learning_rate (float, optional): Training learning rate. Default value is 0.01.
		max_iter (int, optional): Maximum iterations for gradient descent. Default value is 100.
		solver (str, optional): Parameter estimation solver. Default value is 'normal'.
		random_seed (int, optional): Random seed for parameter initialization. Default value is None.
		verbose (bool, optional): Enable verbose output. Default value is False.
	'''

	def __init__(
		self,
		intercept: Optional[bool] = True,
		learning_rate: Optional[float] = 1e-2,
		max_iter: Optional[int] = 100,
		solver: Optional[Literal['gd', 'normal']] = 'normal',
		random_seed: Optional[int] = None,
		verbose: Optional[bool] = False
	) -> None:
		super(LinearRegression, self).__init__()
		assert solver in ['gd', 'normal'], 'Unknown solver'

		self.intercept = intercept
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.solver = solver
		self.random_seed = random_seed
		self.verbose = verbose

		self.w = None

	def _solve_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
		'''
		Estimates best parameters with gradient descent.

		Args:
			X (np.ndarray): The training set features.
			y (np.ndarray): The training set targets.
		'''

		num_samples, num_features = X.shape
		self.w = np.random.randn(num_features + self.intercept)
		for i in range(self.max_iter):
			y_hat = self._predict(X)
			grad = -2 * np.dot(X.T, y - y_hat) / num_samples
			self.w -= self.learning_rate * grad
			if self.verbose:
				error = ((y - y_hat) ** 2).mean()
				print(f'Step {i + 1}, Error {error:.4f}')

	def _solve_normal(self, X: np.ndarray, y: np.ndarray) -> None:
		'''
		Estimates best parameters with normal equation.

		Args:
			X (np.ndarray): The training set features.
			y (np.ndarray): The training set targets.
		'''
		
		if self.intercept:
			ones = np.ones((len(X), 1))
			X = np.hstack([ones, X])
		self.w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))

	def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
		'''
		Args:
			X (np.ndarray): The training set features.
			y (np.ndarray): The training set targets.
		'''

		np.random.seed(self.random_seed)
		if self.solver == 'gd':
			self._solve_gradient_descent(X, y)
		elif self.solver == 'normal':
			self._solve_normal(X, y)

	def _predict(self, X: np.ndarray) -> np.ndarray:
		'''
		Args:
			X (np.ndarray): The set of features.
		'''

		if self.w is None:
			raise Exception()
		if self.intercept:
			ones = np.ones((len(X), 1))
			X = np.hstack([ones, X])
		y_hat = np.dot(X, self.w)
		return y_hat
