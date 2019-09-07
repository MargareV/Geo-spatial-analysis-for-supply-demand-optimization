import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

dataset = np.loadtxt("/home/margs/Data Science Project - Locale/dataset_cat.txt")
x_train, x_test = train_test_split(dataset, test_size=0.3)


kmeans = KMeans(n_clusters=4)
kmeans.fit(dataset)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(dataset[:,0], dataset[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()