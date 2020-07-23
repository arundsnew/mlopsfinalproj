'''
Objective: 
Predicting DoS Attacks, DDoS Attacks by classifying them from Legitimate packets from the Datasets using different Algorithms.
Finding best performing Library / Algorithm and tune for better accuracy if needed using DevOps methodology. 
As a result, to identify best performing Algorithms and corresponding best accuracy achieved during Data Science.
Outcome is that this entire MLOps Architecture will help Data Scientists
'''


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


#Process the datasets
dataset_train = pd.read_csv('dataset/traindata.csv', delimiter=',', nrows = None)
dataset_train.dataframeName = 'traindata.csv'
nRow, nCol = dataset_train.shape
print(f'Training Dataset has {nRow} rows and {nCol} columns')
dataset_test = pd.read_csv('dataset/testdata.csv', delimiter=',', nrows = None)
dataset_test.dataframeName = 'testdata.csv'
nRow, nCol = dataset_test.shape
print(f'Test Dataset has {nRow} rows and {nCol} columns')
label_encoder = preprocessing.LabelEncoder()
dataset_train['Label'] = label_encoder.fit_transform(dataset_train['Label'])
dataset_test['Label'] = label_encoder.fit_transform(dataset_test['Label'])
#Splitting the dataset by Train and Test
X_train = dataset_train.drop('Label',axis=1)
X_test = dataset_test.drop('Label',axis=1)
y_train = dataset_train['Label']
y_test = dataset_test['Label']


#Gaussian Naive Bayes Algorithm
gaussian = GaussianNB()
gaussian.fit(X_train,y_train)
gaussian_pred = gaussian.predict(X_test)
print("Accuracy derived with Gaussian Naive Bayes Algorithm:",metrics.accuracy_score(y_test, gaussian_pred))

#Decision Tree Classifier Algorithm
dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train,y_train)
dt_pred = dtc.predict(X_test)
print("Accuracy derived with Decision Tree Algorithm:",metrics.accuracy_score(y_test, dt_pred))


#K-Nearest Neighbor Classifier Algorithm
knn = KNeighborsClassifier(n_neighbors=2)
knnfit = knn.fit(X_train, y_train)
knn_pred = knnfit.predict(X_test)
print("Accuracy derived with KNN Algorithm:",metrics.accuracy_score(y_test, knn_pred))

# Logistic Regression Classifier Algorithm
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
lr_pred = logisticRegr.predict(X_test)
#score = logisticRegr.score(X_test, y_test)
# OR
accuracy_score(y_test,lr_pred)
print("Accuracy derived with Logistic Regression Algorithm:",metrics.accuracy_score(y_test,lr_pred))


#Random Forest Classifier Algorithm
rfc = RandomForestClassifier()
rfcfit = rfc.fit(X_train, y_train)
rfc_pred1 = rfcfit.predict(X_test)
accuracy_score(y_test,rfc_pred1)
print("Accuracy derived with Randorm Forest Classifier Algorithm:",accuracy_score(y_test,rfc_pred1))


#K-Means Clustering Algorithm
model = KMeans(n_clusters=4)
sc = StandardScaler()
data_scaled = sc.fit_transform(dataset_train)
model.fit(data_scaled)
km_pred  = model.fit_predict(data_scaled)
print("Accuracy derived with K-Means Algorithm:",metrics.accuracy_score(y_test,km_pred))

