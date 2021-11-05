import pandas
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pyplot

data = pandas.read_csv("dataset.csv")

print(data) # we see that the code had an extra unneeded column 

print(data.columns)
data.drop('Unnamed: 0', axis=1, inplace = True) # dropping the first element of the axis = column 
print(data)

data = data.rename({0:'index', 'x1':'data1', 'x2':'data2'}, inplace= True)

data = data.values

pyplot.scatter(data[:,1], data[:,2])
pyplot.savefig("scatter.png") # So we see that the distribution inside the group is different -> KMeans will not work as well

data = data[:,1:3] #inclusive, exclusive
print(data)

# Using the other model (KMeans), just to further compare:
kmean_results = KMeans(n_cluster = 3).fit_predict(data)
print(kmean_results)
pyplot.scatter(data[:,1], data[:,2], c= kmean_results)
pyplot.savefig("scatter_kmeans_3.png") # According to centroid
pyplot.close()
# kmeans : problems when overlapping clusters

# GAUSSIAN MIXTURE MODEL (similar syntax)
gmm_results = GaussianMixture(n_components = 3).fit_predict(data)
print(kmean_results)
pyplot.scatter(data[:,1], data[:,2], c= kmean_results)
pyplot.savefig("scatter_kmeans_3.png")
pyplot.close()
# very dense at the center, and distributino is symetric
# cluster : dense in the center, symetric distribution , even when it is overlapping cluster