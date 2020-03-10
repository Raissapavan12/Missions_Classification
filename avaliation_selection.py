#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('docs/activitiesV2.csv')
df = df.dropna()
df.head()
df.shape

#get data correctly
#df[]

cut = len(df.columns) - 1

X_train = df.iloc[0:4700, 5:cut]
y = df.iloc[0:4700, -1:]

df_new = df.iloc[4700:, :]
X_teste = df_new.iloc[0:, 5:cut]
X_teste.head()

neigh = KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree', weights = 'distance')
neigh.fit(X_train, y)
 
df_new['prediction'] = neigh.predict(X_teste)
df_new.shape

neigh.score(X_train,y)

#a = np.array(df_new.iloc[:,13:14].values)
# b = np.array(df_new.iloc[:,14:].values)

#compare reality and model
erro_perc = ((len(df_new) - len(df_new[df_new['marker'] == df_new['prediction']]))*100)/len(df_new)
print(erro_perc)

df_new.to_excel('docs/output.xlsx')
df.to_excel('docs/original.xlsx')

###
fig, ax = plt.subplots(1,2)
sns.countplot(x='marker', data = df_new, ax=ax[0]).set_title('Real')
sns.countplot(x='prediction', data = df_new, ax=ax[1]).set_title('Previsão')
fig.show()