from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


dataset = pd.read_csv('../input/Mall_Customers.csv')
X= dataset.iloc[:, [3,4]].values

labels = range(1, 201)  
plt.figure(figsize=(25, 20))  
plt.subplots_adjust(bottom=0.1)  
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):  
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

linked = linkage(X, 'single')

labelList = range(1, 201)

plt.figure(figsize=(25, 20))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.savefig('../graph/hierarchical03_dendrigram.png')
plt.show() 


cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')  
cluster.fit_predict(X)


plt.figure(figsize=(25, 20))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.savefig('../graph/hierarchical03_clusters_8.png')
plt.show
