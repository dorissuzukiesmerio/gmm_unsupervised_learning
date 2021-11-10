import pandas
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pyplot


from sklearn.metrics import silhouette_score

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
pyplot.close()

# Using the other model (KMeans), just to further compare:

def run_kmeans(n, data):
	kmeans_machine = KMeans(n_clusters = n) # instead of having this code on the following line, now we have a machine
	kmeans_results = kmeans_machine.fit_predict(data)
	silhouette = 0
	if n>1 :
		silhouette = silhouette_score(data, kmeans_machine.labels_, metric = 'euclidean')
	print(kmeans_results)
	pyplot.scatter(data[:,0], data[:,1], c= kmeans_results)
	pyplot.savefig("scatter_kmeans_"+ str(n) + ".png") # According to centroid
	pyplot.close()
	return silhouette
# kmeans : problems when overlapping clusters

# GAUSSIAN MIXTURE MODEL (similar syntax)

def run_gmm(n, data):
	gmm_machine = GaussianMixture(n_components = n) # instead of having this code on the following line, now we have a machine
	gmm_results = gmm_machine.fit_predict(data)
	silhouette = 0
	if n>1 :
		silhouette = silhouette_score(data, gmm_results, metric = 'euclidean')
	print(gmm_results)
	pyplot.scatter(data[:,0], data[:,1], c= gmm_results)
	pyplot.savefig("scatter_gmm_"+ str(n) + ".png") # According to centroid
	pyplot.close()
	return silhouette



# very dense at the center, and distributino is symetric
# cluster : dense in the center, symetric distribution , even when it is overlapping cluster


# kmeans_silhouette = run_kmeans(3, data)
# gmm_silhouette = run_gmm(3, data)

# print(kmeans_silhouette)
# print(gmm_results)

# Optimal number of clusters:
kmeans_silhouette_list = [run_kmeans(i+1, data) for i in range(7)]
# max_kmeans_silhouette = max(kmeans_silhouette_list)
# print(kmeans_silhouette_score)
print(kmeans_silhouette_list)

gmm_silhouette_list = [run_gmm(i+1, data) for i in range(7)]
print("\nnumber of cluster with max gmm silhouette scores: \n", gmm_silhouette_list.index(max(gmm_silhouette_list))+1)
