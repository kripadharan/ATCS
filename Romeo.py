__author__ = 'Kripa Dharan'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
pd.set_option("display.max_columns",None)


def getCA_UFOs():
    ufos = pd.read_csv("ufos.csv", index_col=False, parse_dates=[1])
    ufos.info()
    ufos = ufos[ ufos['state'] == 'ca' ]
    ufos['shape']=ufos['shape'].fillna('unknown')
    ufos['duration'] = pd.to_numeric(ufos['duration (seconds)'], errors='coerce')
    ufos['latitude'] = pd.to_numeric(ufos['latitude'], errors='coerce')
    ufos['datetime'] = pd.to_datetime(ufos['datetime'],format="%m/%d/%Y %H:%M",errors='coerce')
    ufos = ufos.dropna()
    #print(len(ufos['city'].value_counts()))
    ufos['year'] = ufos['datetime'].dt.year
    ufos['month'] = ufos['datetime'].dt.month
    ufos['hour'] = ufos['datetime'].dt.hour
    ufos = ufos.drop(['country','state','city','date posted','comments','duration (hours/min)','duration (seconds)', 'datetime'], axis=1)
    ufos = pd.concat([ufos, pd.get_dummies(ufos['shape'])], axis=1)
    ufos = ufos.drop(['shape'],axis=1)
    ufos.info()
    return ufos
    


ufos_data = getCA_UFOs()
# Normalize all the data
scaler = StandardScaler()
X = scaler.fit_transform(ufos_data)

from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def exploreEPS(X):
    for num in range(15,35,1):
    #for num in range(3,35):
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = DBSCAN(eps=num/10)
        cluster_labels = clusterer.fit_predict(X)
        num_clusters = len(np.unique(cluster_labels))-1

        if num_clusters > 1:
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            calhara = calinski_harabasz_score(X, cluster_labels)
            db = davies_bouldin_score(X, cluster_labels)
            print("For eps={0:2.2f}, {1:2d} clusters, with silhouette: {2:2.3f} calinski-harabasz: {3:2.3f} davies-bouldin: {4:2.3f}".format(num/10,num_clusters,silhouette_avg,calhara,db))
        else:
            print("For eps={0:2.2f}, only 1 cluster found.".format(num/10))


def exploreClustersDescribe(dataset, X, test_eps, alg):
    algorithm = alg(eps=test_eps)
    Y = algorithm.fit_predict(X)

    print("Number of inputs: " + str(len(X)))
    total = 0 
    clusters = []
    for foo in np.unique(Y):
        if foo != -1:
            clusters.append(dataset[Y==foo])
            print("Now collecting: ",foo,", num= ",len(clusters[foo]))
            total += len(clusters[foo])

    print("Total in clusters: ",total)


    for i in range(len(clusters)):
        print("\n\n--- Looking at cluster ",i,' ---\n')
        print("MODE: \n",clusters[i].mode())
        print("STATS: \n",clusters[i].describe())

exploreEPS(X)
#The eps value that produced the best metrics (highest silhouette and calinski scores and lowest davies score) was 3.3

#information about the clusters with the test eps for DBSCAN
exploreClustersDescribe(ufos_data, X, 3.2, DBSCAN)

#information about the clusters with the test eps for OPTICS
exploreClustersDescribe(ufos_data, X, 3.2, OPTICS)




