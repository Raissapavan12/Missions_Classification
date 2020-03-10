from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #module for principal component analysis

#Do component analysis:
#https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python

df = pd.read_csv('docs/V11.csv')
df = df.fillna(0)
df.head()

df2 = pd.read_csv('docs/V10.csv')
df2 = df2.fillna(0)
df2.head()

#creating label encoder
le = preprocessing.LabelEncoder()
df['companyName'] = le.fit_transform(df['companyName'].astype(str))
df['establishmentName'] = le.fit_transform(df['establishmentName'].astype(str))
df.head()

df2['companyName'] = le.fit_transform(df2['companyName'].astype(str))
df2['establishmentName'] = le.fit_transform(df2['establishmentName'].astype(str))
df2.head()

cut = len(df.columns) - 2

#df treino e teste
X = df.iloc[0:, 3:cut]
y = np.array(df.iloc[0:, -1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

X_train.head()

#df teste
#df_new = df2.sample(n=800)
#df_new.head()

#X_teste = df_new.iloc[0:, 3:cut]
#y_teste = np.array(df_new.iloc[0:,-1:])

#X_teste.head()
#df_new.shape

#Model#
def KNeighbors(X_treino, y_treino, X_teste):
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree', weights = 'distance')
    neigh.fit(X_treino, y_treino)
    X_teste['prediction'] = neigh.predict(X_teste)

    print('Mean acuracy of the model is {}'.format(neigh.score(X_treino,y_treino)))
    return X_teste

KNeighbors(X_train, y_train, X_test)


....
#compare reality and model
erro_perc = ((len(df_new) - len(df_new[df_new['marker'] == df_new['prediction']]))*100)/len(df_new)
print(erro_perc)

#Reverse label encoder -> ARRUMAR
df_new['companyName'] = le.inverse_transform(df_new['companyName'])
df_new['establishmentName'] = le.inverse_transform(df_new['establishmentName'])
df.head()

df_new[df_new['prediction']==1]

#notas 0-2 = 1, notas > 2 = 0
fig, ax = plt.subplots(1,2)
sns.countplot(x='marker', data = df_new, ax=ax[0]).set_title('Real')
sns.countplot(x='prediction', data = df_new, ax=ax[1]).set_title('Previsão')
fig.show()

KNeighbors(X_train, y_train, df_new, X_teste)

df_new.to_excel('docs/output_Kneighbors.xlsx')

#https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn