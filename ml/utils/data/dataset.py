import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def make_regression(
	num_samples: Optional[int] = 100,
	num_features: Optional[int] = 1,
	a: Optional[float] = None,
	b: Optional[float] = 0,
	noise: Optional[float] = 0,
	random_seed: Optional[int] = None 
) -> tuple[np.ndarray, np.ndarray]:
	np.random.seed(random_seed)
	if a is None:
		a = np.random.randn()
	X = np.arange(num_samples, dtype=np.float32).reshape((num_samples, 1))
	X = np.tile(X, (1, num_features))
	if noise > 0:
		X_noise = np.random.normal(0, noise, X.shape)
		X += X_noise
	a_vec = np.full((num_features,), a, dtype=np.float32)
	y = X @ a_vec + b
	if noise > 0:
		y_noise = np.random.normal(0, noise, y.shape)
		y += y_noise
	return X, y

def make_circles(
	num_circles: Optional[int] = 2,
	num_samples: Optional[int] = 100,
	noise: Optional[float] = 0,
	random_seed: Optional[int] = None 
) -> tuple[np.ndarray, np.ndarray]:
	np.random.seed(random_seed)
	points = np.empty((0, 2))
	r = np.random.randn()
	for i in range(num_circles):
		theta = np.random.uniform(0, 2 * np.pi, num_samples)
		x = r / (i + 1) * np.cos(theta)
		y = r / (i + 1) * np.sin(theta)
		if noise > 0:
			x_noise = np.random.normal(0, noise, num_samples)
			y_noise = np.random.normal(0, noise, num_samples)
			x += x_noise
			y += y_noise
		new_points = np.stack((x, y), axis=1)
		points = np.vstack((points, new_points))
	return points