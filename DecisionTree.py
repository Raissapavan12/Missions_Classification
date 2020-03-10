from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('docs/V11.csv')
df = df.fillna(0)
df.head()

#creating label encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['companyName'] = le.fit_transform(df['companyName'].astype(str))
df['establishmentName'] = le.fit_transform(df['establishmentName'].astype(str))
df.head()

cut = len(df.columns) - 2

X = df.iloc[0:, 3:cut]
y = np.array(df.iloc[0:, -1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

print('O dataset de treino possui {} missões e o de treino {} missões.'.format(X_train.shape[0], X_test.shape[0]))

cv = StratifiedKFold(n_splits = 4, shuffle = True)

model = DecisionTreeClassifier(max_depth=3)
results = cross_val_score(model, X_train, y_train, cv = 4, scoring = 'precision')

def intervalo(results):
    mean = results.mean()
    dv = results.std()
    print('Precisão média: {:.2f}%'.format(mean*100))
    print('Intervalo de precisão: [{:.2f}% ~ {:.2f}%]'
           .format((mean - 2*dv)*100, (mean + 2*dv)*100))

intervalo(results)