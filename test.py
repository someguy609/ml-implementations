import numpy as np
from ml.models.linear_regression import LinearRegression

model = LinearRegression(solver='normal', verbose=True, random_seed=42)
X = np.arange(10).reshape(10, 1)
y = 2 * X + 1

model.fit(X, y)
print(model.predict(X))
print(model.w)