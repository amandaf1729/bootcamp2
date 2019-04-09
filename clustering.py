import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

weather_data = 
{'Temperature': [7.74, 9.25, 4.55, 0.94, 7.715, 10.04, 21.825, 25.025, 12.985, 12.690000000000001, 22.51, 20.564999999999998, 26.545, 23.165, 26.29, 23.27, 21.82, 23.71, 25.41, 21.625, 22.63, 25.515, 20.805, 23.17, 25.4, 24.705, 16.8, 22.994999999999997, 22.57, 14.135, 18.055, 16.205, 21.055, 15.315, 11.39, 18.85, 2.645, 20.119999999999997, 12.68, 26.13, 29.1, 31.525, 20.135, 16.39, 28.35, 22.71, 28.405, 26.365000000000002, 28.845, 30.11, 22.795, 9.059999999999999, 15.31, 18.4, 20.0, 24.2, 20.29, 11.79, 12.99, 14.12, 14.835, 22.185000000000002, 13.629999999999999, 24.53, 14.759999999999998, 10.81, 15.48, 17.384999999999998, 28.475, 8.685, 8.185, 26.54, 11.045, 13.125, 13.335, 14.719999999999999], 
'Humidity': [87, 95, 89, 92, 85, 77, 62, 54, 74, 76, 59, 83, 54, 86, 72, 88, 57, 72, 68, 49, 60, 80, 87, 74, 67, 60, 78, 79, 76, 49, 59, 52, 52, 48, 59, 51, 72, 60, 53, 26, 21, 17, 19, 53, 47, 61, 52, 14, 43, 49, 61, 63, 69, 50, 66, 62, 56, 76, 81, 44, 57, 52, 74, 54, 69, 37, 36, 29, 63, 76, 77, 22, 76, 54, 76, 39], 
'Wind Speed': [5.688178273167369, 4.256080887616321, 5.5166271309313615, 3.022340928974787, 4.660629937646809, 1.5463835202417877, 4.1682743104138495, 6.356576870025717, 4.814031561738116, 5.139042687369761, 3.149385121996872, 4.1126756378933695, 2.4173575636787827, 8.136961927757893, 10.003034581794322, 2.3763161367533607, 5.760150561677139, 9.660104657856783, 6.650408826464116, 5.233679195657361, 2.8188047800843075, 3.931978256623983, 4.151374897505236, 4.116903998244916, 6.23950461746865, 4.277151654036428, 3.868246507394152, 5.9501053168827385, 6.7267319944245605, 5.447050252156739, 6.852426113001406, 3.300145493130783, 7.208142457117103, 9.398261177388054, 7.681313746920651, 6.337541427696917, 8.277416391776027, 6.457457070733583, 8.994608267284393, 5.04530362215064, 5.393356180869058, 4.628981630345449, 7.045320790450435, 4.1918066491688535, 6.181467470025717, 3.165243194331769, 4.551513895071449, 7.1944691052069265, 5.664886543100294, 2.915715943161272, 3.850894390604205, 3.528815773567698, 4.66325817227392, 12.184528963464038, 6.11, 8.108677293130784, 6.1081513090882575, 6.110308796627694, 7.436534836554522, 4.763792452079854, 8.465394535305814, 7.188174834165806, 13.22757429615048, 8.653125064171903, 9.09879333108513, 3.203600970333254, 5.856301055483974, 4.888911406747263, 9.926153456809566, 9.420959782101859, 8.425715411614837, 11.470263423783768, 6.851965021843104, 7.2287216804687295, 2.096358575847716, 8.00534817399984], 
'Region': ['New England', 'New England', 'New England', 'New England', 'New England', 'New England', 'Middle Atlantic', 'Middle Atlantic', 'Middle Atlantic', 'Middle Atlantic', 'Middle Atlantic', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'South', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'Midwest', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'South West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West']}

# Get the data we obtained from the API
dataset = pd.DataFrame(weather_data)
dataset['Id'] = range(76)

# Ways to visualize the data:
rows, cols = dataset.shape

# Grouping Data By Region
dataset.groupby('Region').size()

# splitting up the labels and the values for each species:
feature_columns = ['Temperature', 'Humidity', 'Wind Speed']
X = dataset[feature_columns].values
Y = dataset['Region'].values

# Encoding Labels (Turning string species names into integers)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

# Data Visualization:
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(15,10))
parallel_coordinates(dataset.drop("Id", axis=1), "Region")
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()

# Training the model:

# Splitting into training and test datasets:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Creating the learning model
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model with the training data
knn_classifier.fit(X_train, Y_train)

# Making predictions with the test data
Y_pred = knn_classifier.predict(X_test)

# Finding Accuracy:
accuracy = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of model: ' + str(round(accuracy, 2)) + ' %.')

# Testing out different k values

# creating list of K for KNN
k_list = list(range(1,45,2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Displaying results visually
plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
plt.plot(k_list, cv_scores)

plt.show()