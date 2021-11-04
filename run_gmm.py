import pandas
from sklearn.cluster import KMeans
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