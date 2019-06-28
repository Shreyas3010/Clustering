
#depedencies

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

#load train and test datasets

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#filling missing numeric values with mean value.
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

#age

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

#label encoder

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])


y = np.array(train['Survived'])
train= train.drop(['Survived','PassengerId'],axis=1)
test= test.drop(['PassengerId'],axis=1)
#X = np.array(train.drop(['Survived'], 1).astype(float))
X=train.values[:, 0:6]



#linked = linkage(X, 'single')
#
#labelList = range(1, 891)
#
#plt.figure(figsize=(25, 20))  
#dendrogram(linked,  
#            orientation='top',
#            labels=labelList,
#            distance_sort='descending',
#            show_leaf_counts=True)
##plt.savefig('hierarchical03_dendrigram.png')
#plt.show() 


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
y_train_res=cluster.fit_predict(X)

correct=0
for i in range(len(X)):
    if(y[i]==y_train_res[i]):
        correct=correct+1

performance=correct/len(X)
print(performance)

performance=str(performance)

df_cols=train.columns.values
 
plt.figure(figsize=(25, 20))  
plt.scatter(X[:,0], X[:,2], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[2])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_0_2_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(X[:,0], X[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[3])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_0_3_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X[:,0], X[:,4], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[4])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_0_4_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X[:,0], X[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[5])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_0_5_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(X[:,2], X[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[3])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_2_3_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X[:,2], X[:,4], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[4])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_2_4_clusters_2.png')
plt.show(0)

plt.figure(figsize=(25, 20))  
plt.scatter(X[:,2], X[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[5])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_2_5_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(X[:,3], X[:,4], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[3])
plt.ylabel(df_cols[4])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_3_4_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X[:,3], X[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[3])
plt.ylabel(df_cols[5])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_3_5_clusters_2.png')
plt.show(0)

plt.figure(figsize=(25, 20))  
plt.scatter(X[:,4], X[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[4])
plt.ylabel(df_cols[5])
plt.title('Train Data Train (Cluster=2)')
plt.suptitle('Performance : '+performance)
plt.savefig('../graph/hierarchical01_train_4_5_clusters_2.png')
plt.show()



#scaling

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
y_train_res_scaled=cluster.fit_predict(X_scaled)

correct=0
for i in range(len(X_scaled)):
    if(y[i]==y_train_res_scaled[i]):
        correct=correct+1

performance_afterscaling=correct/len(X_scaled)
print(performance_afterscaling)

performance_afterscaling=str(performance_afterscaling)


df_cols=train.columns.values
 
plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,0], X_scaled[:,2], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[2])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_0_2_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,0], X_scaled[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[3])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_scaled_train_0_3_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,0], X_scaled[:,4], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[4])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_0_4_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,0], X_scaled[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[5])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_0_5_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,2], X_scaled[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[3])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_2_3_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,2], X_scaled[:,4], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[4])
plt.title('Train Data Train Scaled (Cluster=2)')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_2_4_clusters_2.png')
plt.show(0)

plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,2], X_scaled[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[5])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_2_5_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,3], X_scaled[:,4], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[3])
plt.ylabel(df_cols[4])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_3_4_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,3], X_scaled[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[3])
plt.ylabel(df_cols[5])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_3_5_clusters_2.png')
plt.show(0)

plt.figure(figsize=(25, 20))  
plt.scatter(X_scaled[:,4], X_scaled[:,5], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[4])
plt.ylabel(df_cols[5])
plt.title('Train Data Train Scaled (Cluster=2) ')
plt.suptitle('Performance : '+performance_afterscaling)
plt.savefig('../graph/hierarchical01_train_scaled_4_5_clusters_2.png')
plt.show()
