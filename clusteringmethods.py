import time                   
import warnings               

import numpy as np            
import pandas as pd           
import matplotlib.pyplot as plt                   

from sklearn import cluster, mixture             
from sklearn.preprocessing import StandardScaler  

import os                  
import sys 

from sklearn import metrics

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# %matplotlib inline
warnings.filterwarnings('ignore')

import seaborn as sns


#X= pd.read_csv("../input/2017.csv", header = 0)
#X= pd.read_csv("../input/2016.csv", header = 0)
X= pd.read_csv("../input/2015.csv", header = 0)
#X=X.iloc[:,2:]
X=X.iloc[:,3:]
#X_cols=X.columns.values
#X=pd.DataFrame(data=StandardScaler().fit_transform(X),columns=X_cols)

#result
cl_dist = {'Name' : ['KMeans','MeanShift','MiniBatchKMeans','SpectralClustering','DBSCAN','AffinityPropagation','Birch','GaussianMixture']}
cl_df = pd.DataFrame(cl_dist)
#cl=pd.Series(['KMeans','MeanShift','MiniBatchKMeans','SpectralClustering','DBSCAN','AffinityPropagation','Birch','GaussianMixture'])


#KMeans
result_KMeans = cluster.KMeans(n_clusters= 2).fit_predict(X)

cl_df.loc[cl_df.Name == 'KMeans', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_KMeans, metric='euclidean')
cl_df.loc[cl_df.Name == 'KMeans', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_KMeans)


#MeanShift
result_MeanShift = cluster.MeanShift(bandwidth= 0.2).fit_predict(X)

cl_df.loc[cl_df.Name == 'MeanShift', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_MeanShift, metric='euclidean')
cl_df.loc[cl_df.Name == 'MeanShift', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_MeanShift)


#MiniBatchKMeans
result_MiniBatchKMeans = cluster.MiniBatchKMeans(n_clusters= 2).fit_predict(X)

cl_df.loc[cl_df.Name == 'MiniBatchKMeans', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_MiniBatchKMeans, metric='euclidean')
cl_df.loc[cl_df.Name == 'MiniBatchKMeans', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_MiniBatchKMeans)


#SpectralClustering
result_SpectralClustering = cluster.SpectralClustering(n_clusters= 2).fit_predict(X)

cl_df.loc[cl_df.Name == 'SpectralClustering', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_SpectralClustering, metric='euclidean')
cl_df.loc[cl_df.Name == 'SpectralClustering', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_SpectralClustering)


#DBSCAN
result_DBSCAN = cluster.DBSCAN(eps=0.3).fit_predict(X)

cl_df.loc[cl_df.Name == 'DBSCAN', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_DBSCAN, metric='euclidean')
cl_df.loc[cl_df.Name == 'DBSCAN', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_DBSCAN)


#AffinityPropagation
result_AffinityPropagation = cluster.AffinityPropagation(damping=0.9, preference=-200).fit_predict(X)

cl_df.loc[cl_df.Name == 'AffinityPropagation', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_AffinityPropagation, metric='euclidean')
cl_df.loc[cl_df.Name == 'AffinityPropagation', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_AffinityPropagation)


#Birch
result_Birch = cluster.Birch(n_clusters= 2).fit_predict(X)

cl_df.loc[cl_df.Name == 'Birch', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_Birch, metric='euclidean')
cl_df.loc[cl_df.Name == 'Birch', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_Birch)


#GaussianMixture
gmm = mixture.GaussianMixture( n_components=2, covariance_type='full')
gmm.fit(X)
result_GaussianMixture=gmm.predict(X)

cl_df.loc[cl_df.Name == 'GaussianMixture', 'Silhouette-Coeff'] = metrics.silhouette_score(X, result_GaussianMixture, metric='euclidean')
cl_df.loc[cl_df.Name == 'GaussianMixture', 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(X, result_GaussianMixture)






