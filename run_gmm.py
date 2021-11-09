import pandas
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pyplot

import

data = pandas.read_csv("dataset.csv")

# DATA PREP:
print(data) # we see that the code had an extra unneeded column 

print(data.columns)
data.drop('Unnamed: 0', axis=1, inplace = True) # dropping the first element of the axis = column 
print(data) 

#Or:
# data = data.rename({0:'index', 'x1':'data1', 'x2':'data2'}, inplace= True) # we don't want the index. 
# data = data[:,1:3] #inclusive, exclusive

#sklearn-friendly format
data = data.values
print(data)

# VISUALIZING 

pyplot.scatter(data[:,0], data[:,1])
pyplot.savefig("scatter.png") # So we see that the distribution inside the group is different -> KMeans will not work as well


# Using the other model (KMeans), just to further compare:

def run_means(n, data):
	kmeans_machine = KMeans(n_clusters = n)
	kmean_results = kmeans_machine.fit_predict(data)
	print(kmean_results)
	pyplot.scatter(data[:,0], data[:,1], c= kmean_results)
	pyplot.savefig("scatter_kmeans_3.png") # According to centroid
	pyplot.close()
# kmeans : problems when overlapping clusters

# GAUSSIAN MIXTURE MODEL (similar syntax)
gmm_results = GaussianMixture(n_components = 3).fit_predict(data)
print(kmean_results)
pyplot.scatter(data[:,0], data[:,1], c= kmean_results)
pyplot.savefig("scatter_gmm_3.png")
pyplot.close()
# very dense at the center, and distributino is symetric
# cluster : dense in the center, symetric distribution , even when it is overlapping cluster