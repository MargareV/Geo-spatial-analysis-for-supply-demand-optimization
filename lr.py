import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import datasets, linear_model
import random
import tkinter as tk
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import math
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split



dataset = np.loadtxt("/home/margs/Data Science Project - Locale/dataset_cat.txt")
print(dataset.shape)

category = np.loadtxt("/home/margs/Data Science Project - Locale/category_dataset.txt")
print(category.shape)

x_train, x_test, y_train, y_test = train_test_split(dataset, category, test_size=0.3)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
print(y_pred)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print(rms)