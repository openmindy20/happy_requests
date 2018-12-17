# The eigenvector with the largest eigenvalue is the direction along which the data set has the maximum variance.

# If we want for example to reduce a dataset to just one dimension. In genral, this means picking a unit vector u, and replacing each data point, xi, with its projection along this vector uTxi.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# We import the dataset
df = pd.read_csv('iris.csv').drop('class', axis=1)[['petal_len','petal_wid']]

# We standardize the dataset
df = pd.DataFrame(StandardScaler().fit_transform(df))

# We Calculate the Correlation Matrix
corr_df = pd.DataFrame(data = df).corr()

# We Calculate the EigenVectors and EigenValues of the Correlation Matrix
eig_vals, eig_vecs = np.linalg.eig(corr_df)

# We plot the instances of the dataset on the EigenVectors
df = df.dot(eig_vecs)

# We can then take the first few Principal Components of the DataFrame
df = df.iloc[:,0:2]

# The equivalent on Scikit Learn
'''
transformer = PCA(n_components=2)
df = transformer.fit_transform(df)
'''

df








