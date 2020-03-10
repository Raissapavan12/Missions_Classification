from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('docs/workers_data.csv')
df = df.dropna()
df.head()

kmeans = KMeans(n_clusters = 5, random_state = 42)  
kmeans.fit(df[[
'activitiesFinished'
,'avgRateRuptures'
,'avgDurationMinutes'
,'health']]) 
print(kmeans.cluster_centers_)

df['agrupamento'] = kmeans.labels_

df_total = df.groupby('agrupamento').size().reset_index(name="count")
df_total.head()

#x1 = sns.scatterplot(x="avgMixSize", y="avgRateRuptures", data=df2)
#x2 = sns.scatterplot(x="avgRateRuptures", y="agrupamento", data=df2)
#x3 = sns.scatterplot(x="avgDurationMinutes", y="agrupamento", data=df2)

#print(x1)
#print(x2)
#print(x3)

sns.violinplot(x='agrupamento', y='activitiesFinished',data = df)
sns.violinplot(x='agrupamento', y='avgDurationMinutes',data = df)


sns.FacetGrid(col='agrupamento',hue='avgDurationMinutes',data=df,legend_out=False).map(sns.scatterplot, 'avgDurationMinutes', 'health')
sns.FacetGrid(col='agrupamento',hue='avgDurationMinutes',data=df,legend_out=False).map(sns.scatterplot, 'avgRateRuptures', 'agvRateOutStorage')

df.head()



#https://www.kaggle.com/yugagrawal95/k-means-clustering-using-seaborn-visualization