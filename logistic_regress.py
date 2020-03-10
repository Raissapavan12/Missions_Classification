from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('docs/V7.csv')
df = df.fillna(0)
df = df.dropna()
df.head()
df.shape

cut = len(df.columns) - 2

#df teste
df_new = df.sample(n=700)
df_new.head()

X_teste = df_new.iloc[0:, 5:cut]
y_teste = np.array(df_new.iloc[0:,-1:])
X_teste.shape
y_teste.shape

X_teste.head()
X_teste.shape

#df train
df_train = df.sample(700)
X_train = df_train.iloc[0:, 5:cut]
y_train = np.array(df_train.iloc[0:700, -1:])
X_train.head()

##Model##

def select_mission(X_teste, y_teste, df_novo):
    clf = LogisticRegression(random_state=0, solver = 'liblinear').fit(X_train, y_train)
    df_novo['prediction'] = clf.predict(X_teste)
    
    acuracy = clf.score(X_teste, y_teste)

    print('Mean acuracy of the model is {}'.format(np.around(acuracy, decimals = 1)))
    print(df_novo.shape)
    return df_novo

select_mission(X_teste, y_teste, df_new)

#compare reality and model
erro_perc = ((len(df_new) - len(df_new[df_new['marker'] == df_new['prediction']]))*100)/len(df_new)
print(erro_perc)

#notas 0-2 = 1, notas > 2 = 0
fig, ax = plt.subplots(1,2)
sns.countplot(x='marker', data = df_new, ax=ax[0]).set_title('Real')
sns.countplot(x='prediction', data = df_new, ax=ax[1]).set_title('Previsão')
fig.show()

df_new.to_excel('docs/output_log.xlsx')

#confimation dataset
df_ev = pd.read_csv('docs/not_avaliated.csv')
df_ev = df_ev.dropna()
df_ev.head()
df_ev.shape

X_conf = df_ev.iloc[0:, 5:cut]
y_conf = np.array(df_ev.iloc[0:,-1:])
X_conf.shape
y_conf.shape

select_mission(X_conf, y_conf, df_ev)

#export data
df_ev.to_excel('docs/output_log.xlsx')
df.to_excel('docs/original_log.xlsx')