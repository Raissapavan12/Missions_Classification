#https://medium.com/data-hackers/como-avaliar-seu-modelo-de-classifica%C3%A7%C3%A3o-34e6f6011108

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('docs/V12.csv')
df = df.fillna(0)
df.head()
df.shape

df.corr()
df.describe()

#creating label encoder
le = preprocessing.LabelEncoder()
df['companyName'] = le.fit_transform(df['companyName'].astype(str))
df['establishmentName'] = le.fit_transform(df['establishmentName'].astype(str))
df.head()

#Get Dummies
data = pd.get_dummies(df['companyName'])
pd.concat([data, X], axis = 1)

cut = len(df.columns) - 2

#df treino e teste
X = df.iloc[0:, 3:11]
X.head()
#X = df.iloc[0:, 3:cut]
y = np.array(df.iloc[0:, -1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

X_train.head()

cv = StratifiedKFold(n_splits = 4, shuffle = True)

#model#
model =  KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree', weights = 'distance')

#testing scoring metric#
y_pred = cross_val_score(model, X_train, y_train, cv = cv, scoring = 'accuracy') 

def intervalo(results):
    mean = results.mean()
    dv = results.std()
    print('Precisão média: {:.2f}%'.format(mean*100))
    print('Intervalo de precisão: [{:.2f}% ~ {:.2f}%]'
           .format((mean - 2*dv)*100, (mean + 2*dv)*100))

intervalo(y_pred)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model.score(X_test,y_test)

from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, 
            ax=ax, fmt='d', cmap='Reds')
ax.set_title("Matriz de Confusão", fontsize=14)
ax.set_ylabel("True label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()

from sklearn.metrics import classification_report
print('Relatório de classificação:\n', classification_report(y_test, y_predict, digits=4))

#------------------------------------------#------------------------------------------#

### Curva ROC?
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, score)

fig, ax = plt.subplots(figsize = (12,4))
plt.plot(fpr, tpr, linewidth=2, label = 'KNeighbor')
plt.plot([0,1], [0,1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc = 'lower right')
plt.title('Curva ROC', fontsize = 14)
plt.show()

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html