import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

dataset = np.loadtxt("/home/margs/Data Science Project - Locale/dataset_cat.txt")
print(dataset.shape)

category = np.loadtxt("/home/margs/Data Science Project - Locale/category_dataset.txt")
print(category.shape)

category = np.reshape(category, (41700, 1))
dataset_new = np.concatenate((dataset, category), axis=1)
print(dataset_new.shape)

#real_data = pd.DataFrame(data=dataset_new, columns=['aread_id', 'timestamp', 'category'])

#sns.heatmap(real_data.corr(), annot=True)
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(dataset, category, test_size=0.3)

model = Sequential()

#Hidden Layer
model.add(Dense(2, activation='relu', kernel_initializer='random_normal', input_dim=2))

#output layer
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
model.fit(x_train, y_train, batch_size=400, epochs=50)

eval_model = model.evaluate(x_train, y_train)
print(eval_model)

y_pred = model.predict(x_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred.round())
print(cm)