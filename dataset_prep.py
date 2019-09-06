import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import seaborn as sns



area_id = np.loadtxt("/home/margs/Data Science Project - Locale/Data locale - from area id.txt", dtype='int')
area_id = area_id[0:41700]
print(area_id.shape)

timestamp = np.loadtxt("/home/margs/Data Science Project - Locale/timestamp.txt")
timestamp = timestamp[0:41700]
print(timestamp.shape)

category = np.loadtxt("/home/margs/Data Science Project - Locale/category_dataset.txt")
print(category.shape)


timestamp_int = []

for time in timestamp:
	timestamp_int.append(int(time))

timestamp_int = np.array(timestamp_int)
print(timestamp_int.shape)
#np.savetxt("times_int.txt", timestamp_int)
category_int = []

for categories in category:
	category_int.append(int(categories))

category_int = np.array(category_int)
print(category_int.shape)
print(np.bincount(category_int))

timestamp_intbincount = np.bincount(timestamp_int)
print(timestamp_intbincount.shape)

np.savetxt("bincount_area.txt", np.bincount(area_id))
bincount_area = np.loadtxt("/home/margs/Data Science Project - Locale/bincount_area.txt")

print(bincount_area.shape)
max_count = np.amax(bincount_area)
print(max_count)

for i in range(len(bincount_area)):
	if(bincount_area[i]==max_count):
		print(i)

category_area = []
for area in area_id:
	if(area==393):
		category_area.append(2)
	elif(area==571):
		category_area.append(3)
	elif(area==293):
		category_area.append(4)
	else:
		category_area.append(0)
category_area = np.array(category_area)
print(category_area.shape)
print(np.bincount(category_area))

for k in range(len(area_id)):
	if(category_area[k]==4 & category_int[k]==1):
		category_int[k] = 4
print(np.bincount(category_int))


#for i in range(len(area_id)):
#	if(area_id[i]==392 & category_int[i]==1):
#		category_int[i] = 2
	#elif(area_id[i]==569 & category_int[i]==1):
	#	category_int[i] = 3
	#elif(area_id[i]==292 & category_int[i]==1):
	#	category_int[i] = 4
#print(category_int.shape)
#print(np.bincount(category_int))

#category = []

#for times in timestamp_int:
#	if(times==7):
#		category.append(1)
#	elif(times==8):
#		category.append(1)
#	elif(times==9):
#		category.append(1)
#	elif(times==17):
#		category.append(1)
#	if(times==18):
#		category.append(1)
#	else:
#		category.append(0)

#category = np.array(category)
#print(category.shape)
#print(np.bincount(category))
#np.savetxt("category_dataset.txt", category)
#dataset_new = np.reshape(dataset, (21715, 2))
#print(dataset_new.shape)

#arr = [[0, 0, 1], [0, 1, 1.1], [1, 0, 1], [1, 1, 2]]
#arr = np.array(arr)
#print(arr.shape)
#print(arr)

#area_id = np.reshape(area_id, (41700, 1))
#timestamp = np.reshape(timestamp, (41700, 1))

#dataset = np.concatenate((area_id, timestamp), axis=1)
#print(dataset.shape)
#np.savetxt("dataset_cat.txt", dataset)
