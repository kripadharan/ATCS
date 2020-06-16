__author__ = 'Kripa Dharan'
"""
In this file, I trained different models to predict whether a mushroom would be edible or poisonous
given a variety of different features, including cap-shape, cap-color, etc. I encoded the data in 
three different ways: One Hot Encoding, Ordinal Encoding, Binary Encoding. I also ran three different
models for each way of encoding: Logistic Regression, Decision Tree Classifier, and K Nearest Neighbors
Classifier.
"""
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

#mushrooms dataset
mushrooms = pd.read_csv("mushrooms.csv")

#naming the columns
mushrooms.columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", 
	"gill-spacing", "gill-size","gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
	"stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", 
	"spore-print-color", "population", "habitat"]

mushrooms.info()
#splitting the dataset into training and testing data with a ratio of 80/20 - ratio given by the assignment
sp = StratifiedShuffleSplit(n_splits = 1, train_size = 0.8, test_size = 0.2, random_state = 0)
for train_index, test_index in sp.split(mushrooms[mushrooms.columns[1:]], mushrooms['class']):
	mushrooms_train, mushrooms_test = mushrooms.loc[train_index], mushrooms.loc[test_index]

def genDataOneHot(train, test):
	"""  encode the data using scikit One Hot Encoding - returns 2 pandas dataframes, inputs 2 pandas dataframes
		This modifies the original dataframes that get passed in
	"""
	print("*********************************************************************************************")
	enc = OneHotEncoder(sparse = False)
	train = enc.fit_transform(train)
	test = enc.transform(test)

	train = pd.DataFrame(train)
	test = pd.DataFrame(test)

	print("Total Columns After One Hot Encoding: " + str(len(train.columns)))

	return train, test

def genDataOrdinal(train, test):
	""" encode the data using scikit Ordinal Encoding - returns 2 pandas dataframes, inputs 2 pandas dataframes
		This modifies the original dataframes that get passed in
	"""
	print("*********************************************************************************************")
	enc = OrdinalEncoder()
	train = enc.fit_transform(train)
	test = enc.transform(test)

	train = pd.DataFrame(train)
	test = pd.DataFrame(test)

	print("Total Columns After Ordinal Encoding: " + str(len(train.columns)))

	return train, test

def genDataBinary(train, test):
	""" encode the data using Binary Encoding - returns 2 pandas dataframes, inputs 2 pandas dataframes
		to encode the data in binary, I first encoded using scikit ordinal encoding.
		Then, I created a new column for each necessary binary digit of the max ordinal value in each column,
		and coverted each ordinal value to binary, placing each digit in a new column. Finally, I dropped
		the original columns with the ordinal values from the dataframe.

		This modifies the original dataframes that get passed in
	"""
	print("*********************************************************************************************")
	train, test = genDataOrdinal(train, test)
	totalcolumns = 0
	for col in range(1, len(train.columns)):
		maxORD = train[train.columns[col]].max()
		if not maxORD == 0:
			addCols = int(math.log(maxORD, 2) + 1)
		else:
			adddCols = 0
		for i in range(addCols):
			train[(str(col) + "-" + str(addCols - i))] = train[train.columns[col]] // (2**(addCols - i - 1))
			train[train.columns[col]] = train[train.columns[col]] % (2**(addCols - i - 1))

			test[(str(col) + "-" + str(addCols - i))] = test[test.columns[col]] // (2**(addCols - i - 1))
			test[test.columns[col]] = test[test.columns[col]] % (2**(addCols - i - 1))

	dropCols = list(range(1, 23))
	train = train.drop(dropCols, axis = 1)
	test = test.drop(dropCols, axis = 1)

	print("Total Columns After Binary Encoding: " + str(len(train.columns)))

	return train, test

def getInputsTargets(train, test):
	""" generate inputs and targets given the data - returns 4 pandas dataframes, inputs 2 pandas dataframes 
		takes the first column of the train and test and returns those as targets
		the rest of the columns are returned as the inputs
	"""
	train_targets = train[train.columns[:1]]
	test_targets = test[test.columns[:1]]

	train_inputs = train[train.columns[1:]]
	test_inputs = test[test.columns[1:]]
	return train_inputs, train_targets, test_inputs, test_targets

def logisticReg(train_inputs, train_targets, test_inputs, test_targets, encoder):
	""" run a scikit logistic regression for train and test data - prints mean accuracy and confusion matrix
	"""
	print("Running Logistic Regression with " + encoder + "...")
	log_reg = LogisticRegression(penalty="none", solver="saga")
	start = time.perf_counter()
	log_reg.fit(train_inputs, train_targets)
	log_reg_outputs = log_reg.predict(train_inputs)
	stop = time.perf_counter()
	elapsed = stop - start

	print("Mean train accuracy:")
	print(log_reg.score(train_inputs, train_targets))
	print("Confusion Matrix:")
	print(confusion_matrix(train_targets, log_reg_outputs))
	print("Elapsed Time: " + str(elapsed) + " seconds")

	log_reg_outputs = log_reg.predict(test_inputs)
	print("Mean test accuracy:")
	print(log_reg.score(test_inputs, test_targets))
	print("Confusion Matrix:")
	print(confusion_matrix(test_targets, log_reg_outputs))
	print("*********************************************************************************************")

def decTreeClassify(train_inputs, train_targets, test_inputs, test_targets, encoder):
	""" run a scikit decision tree classifier for train and test data - prints mean accuracy and confusion matrix
	"""
	print("Running Decision Tree fit with " + encoder + "...")	
	tree = DecisionTreeClassifier()
	start = time.perf_counter()
	tree.fit(train_inputs, train_targets)
	train_outputs = tree.predict(train_inputs)
	stop = time.perf_counter()
	elapsed = stop - start	

	print("Mean train accuracy: ", tree.score(train_inputs, train_targets))
	print("Confusion Matrix (train):")
	print(confusion_matrix(train_targets, train_outputs))
	print("Elapsed Time: " + str(elapsed) + " seconds")

	print("Mean test accuracy: ", tree.score(test_inputs, test_targets))
	test_outputs = tree.predict(test_inputs)
	print("Confusion Matrix (test):")
	print(confusion_matrix(test_targets, test_outputs))
	print("*********************************************************************************************")

def knnclassify(train_inputs, train_targets, test_inputs, test_targets, encoder):
	""" run a scikit K nearest neighbors classifier for train and test data - prints mean accuracy and confusion matrix
	"""
	print("Running K Nearest Neighbors classifier with " + encoder + "...")
	knn = KNeighborsClassifier() 

	start = time.perf_counter()
	knn.fit(train_inputs, train_targets)
	outputs = knn.predict(train_inputs)
	stop = time.perf_counter()
	elapsed = stop - start

	print("Mean train accuracy:", knn.score(train_inputs, train_targets))
	print(confusion_matrix(train_targets, outputs))
	print("Elapsed Time: " + str(elapsed) + " seconds")

	outputs = knn.predict(test_inputs)
	print("Mean test accuracy:", knn.score(test_inputs, test_targets))
	print(confusion_matrix(test_targets, outputs))
	print("*********************************************************************************************")

#these lists are just used to efficiently call the methods in the following for loop
encodersTypes = [genDataOneHot, genDataOrdinal, genDataBinary]
classifiers = [logisticReg, decTreeClassify, knnclassify]

#this is just so that the ML algorithms print the type of encoding used on the data
encoderStr = ["One Hot Encoding", "Ordinal Encoding", "Binary Encoding"]

for i in range(len(encodersTypes)): 
	print("Predictions with different models using " + encoderStr[i])
	train, test = encodersTypes[i](mushrooms_train, mushrooms_test)
	for j in range(len(classifiers)): 
		train_inputs, train_targets, test_inputs, test_targets = getInputsTargets(train, test)
		classifiers[j](train_inputs, train_targets, test_inputs, test_targets, encoderStr[i])


