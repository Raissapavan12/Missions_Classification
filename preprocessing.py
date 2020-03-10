import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

df = pd.read_csv('docs/V8.csv', sep = ';')
df.head()

le = preprocessing.LabelEncoder()
df['companyName'] = le.fit_transform(df['companyName'].astype(str))
df['establishmentName'] = le.fit_transform(df['establishmentName'].astype(str))
df.head()

df['establishmentName'] = le.inverse_transform(df['establishmentName'])
df['companyName'] = le.inverse_transform(df['companyName'])
df.head()