# Import necessary libraries
from scipy.cluster.hierarchy import dendrogram, linkage  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("../input/Iris.csv") #load the dataset
df.drop('Id',axis=1,inplace=True) # Se elimina la columna no requerida


# Change categorical data to number 0-2
df["Species"] = pd.Categorical(df["Species"])
df["Species"] = df["Species"].cat.codes
# Change dataframe to numpy matrix
data = df.values[:, 0:4]
category = df.values[:, 4]

linked = linkage(data, 'single')

labelList = range(1, 201)

plt.figure(figsize=(25, 20))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.savefig('../graph/hierarchical02_dendrogram.png')
plt.show() 


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data)

df_cols=df.columns.values
 
plt.figure(figsize=(25, 20))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[1])
plt.title('Iris Data (Cluster=2)')
plt.savefig('../graph/hierarchical02_0_1_clusters_2.png')
plt.show

plt.figure(figsize=(25, 20))  
plt.scatter(data[:,0], data[:,2], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[2])
plt.title('Iris Data (Cluster=2)')
plt.savefig('../graph/hierarchical02_0_2_clusters_2.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(data[:,0], data[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[0])
plt.ylabel(df_cols[3])
plt.title('Iris Data (Cluster=2)')
plt.savefig('../graph/hierarchical02_0_3_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(data[:,1], data[:,2], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[1])
plt.ylabel(df_cols[2])
plt.title('Iris Data (Cluster=2)')
plt.savefig('../graph/hierarchical02_1_2_clusters_2.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(data[:,1], data[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[1])
plt.ylabel(df_cols[3])
plt.title('Iris Data (Cluster=2)')
plt.savefig('../graph/hierarchical02_1_3_clusters_2.png')
plt.show(0)

plt.figure(figsize=(25, 20))  
plt.scatter(data[:,2], data[:,3], c=cluster.labels_, cmap='rainbow')  
plt.xlabel(df_cols[2])
plt.ylabel(df_cols[3])
plt.title('Iris Data (Cluster=2)')
plt.savefig('../graph/hierarchical02_2_3_clusters_2.png')
plt.show()
