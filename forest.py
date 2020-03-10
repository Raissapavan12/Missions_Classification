from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import pandas as pd
import numpy as np

df = pd.read_csv('docs/forest_test1.csv')
df = df.dropna()

df.head()

X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))



#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html