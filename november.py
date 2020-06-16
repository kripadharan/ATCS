__author__ = 'Kripa Dharan'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Perceptron

# MNIST data set of handwritten digits
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

#Titanic data set
titanic_train = pd.read_csv("titanic_train2.csv")
titanic_test = pd.read_csv("titanic_test2.csv")

#Pulsar data set
pulsar_train = pd.read_csv("pulsar_train.csv")
pulsar_test = pd.read_csv("pulsar_test.csv")

#Cancer data set
cancer_train = pd.read_csv("cancer_train.csv")
cancer_test = pd.read_csv("cancer_test.csv")

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def decTreeClassify(train_inputs, train_targets, test_inputs, test_targets, depth, samples_leaf):
	tree = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = samples_leaf)
	tree.fit(train_inputs, train_targets)
	print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())

	print("Mean train accuracy: ", tree.score(train_inputs, train_targets))
	train_outputs = tree.predict(train_inputs)
	print("Confusion Matrix (train):")
	print(confusion_matrix(train_targets, train_outputs))


	print("Mean test accuracy: ", tree.score(test_inputs, test_targets))
	test_outputs = tree.predict(test_inputs)
	print("Confusion Matrix (test):")
	print(confusion_matrix(test_targets, test_outputs))

	print("DTC mean accuracies with 4-Fold CV:")
	print( cross_val_score( tree, train_inputs, train_targets, cv=4, scoring="accuracy"))

def logisticReg(train_inputs, train_targets, test_inputs, test_targets):
	log_reg = LogisticRegression(penalty="none", solver="saga")
	log_reg.fit(train_inputs, train_targets)
	log_reg_outputs = log_reg.predict(train_inputs)
	print("Mean train accuracy:")
	print(log_reg.score(train_inputs, train_targets))
	print("Confusion Matrix:")
	print(confusion_matrix(train_targets, log_reg_outputs))


	log_reg_outputs = log_reg.predict(test_inputs)
	print("Mean test accuracy:")
	print(log_reg.score(test_inputs, test_targets))
	print("Confusion Matrix:")
	print(confusion_matrix(test_targets, log_reg_outputs))

def knnclassify(train_inputs, train_targets, test_inputs, test_targets, neighbors, weight):
	print("K Nearest Neighbors classifier")
	knn = KNeighborsClassifier(n_neighbors = neighbors, weights = weight) 
	knn.fit(train_inputs, train_targets)
	outputs = knn.predict(train_inputs)
	print("Mean train accuracy:", knn.score(train_inputs, train_targets))
	print(confusion_matrix(train_targets, outputs))

	outputs = knn.predict(test_inputs)
	print("Mean test accuracy:", knn.score(test_inputs, test_targets))
	print(confusion_matrix(test_targets, outputs))

"""
#1
mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]

# Now let's shuffle the training set to reduce bias opportunities
from sklearn.utils import shuffle
train_targets, train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

decTreeClassify(mnist_train_inputs, mnist_train_targets, mnist_test_inputs, mnist_test_targets, 33, 3)
#I used max_depth = 33 and min_samples_leaf = 3
"""
"""
#2
def gender_to_type(sex):
	if sex == 'female':
		return 1
	else:
		return 0
titanic_train['gender'] = titanic_train['Sex'].apply(gender_to_type)

titanic_inputs = titanic_train[['gender', 'Pclass', 'SibSp', 'Fare']]
titanic_targets = titanic_train['Survived']

titanic_test['gender'] = titanic_test['Sex'].apply(gender_to_type)

titanic_test_inputs = titanic_test[['gender', 'Pclass', 'SibSp', 'Fare']]
titanic_test_targets = titanic_test['Survived']

decTreeClassify(titanic_inputs, titanic_targets, titanic_test_inputs, titanic_test_targets, 33, 3)
#The decision tree performed slightly better than the perceptron after I added more inputs for both models.
#I used max_depth = 33 and min_samples_leaf = 3
"""
"""
#3
pulsar_train_targets = pulsar_train['Pulsar']
pulsar_train_inputs = pulsar_train.drop('Pulsar', axis = 1)

pulsar_test_targets = pulsar_test['Pulsar']
pulsar_test_inputs = pulsar_test.drop('Pulsar', axis = 1)

#logistic regression
logisticReg(pulsar_train_inputs, pulsar_train_targets, pulsar_test_inputs, pulsar_test_targets)

#decision tree
decTreeClassify(pulsar_train_inputs, pulsar_train_targets, pulsar_test_inputs, pulsar_test_targets, 20, 12)

#K nearest neighbors
knnclassify(pulsar_train_inputs, pulsar_train_targets, pulsar_test_inputs, pulsar_test_targets, 9, 'uniform')

#decision tree with max_depth = 20 and min_samples_leaf = 12 did the best, k nearest neighbors with n_neighbors = 9 was second, logistic was the worst

"""
#4
def class_to_targets(target):
		if target == 4:
			return 1
		else:
			return 0
cancer_train_targets = cancer_train['Class'].apply(class_to_targets)
cancer_train_inputs = cancer_train.drop(['Class', 'Bare Nuclei'], axis = 1)

cancer_test_targets = cancer_test['Class'].apply(class_to_targets)
cancer_test_inputs = cancer_test.drop(['Class', 'Bare Nuclei'], axis = 1)

#logistic regression
logisticReg(cancer_train_inputs, cancer_train_targets, cancer_test_inputs, cancer_test_targets)

#decision tree
decTreeClassify(cancer_train_inputs, cancer_train_targets, cancer_test_inputs, cancer_test_targets, 7, 7)

#K nearest neighbors
knnclassify(cancer_train_inputs, cancer_train_targets, cancer_test_inputs, cancer_test_targets, 12, 'uniform')

#decision tree with max_depth = 7 and min_samples_leaf = 7 did the best, k nearest neighbors with n_neighbors = 12 was second, logistic was the worst

