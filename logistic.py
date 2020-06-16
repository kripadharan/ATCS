import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
# Helper functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

def print_class_results(targets, outputs):
	print_conf_matrix(targets, outputs)

	# Precision - How accurate are the positive predictions?
	print("Precision (TP / (TP + FP)):", precision_score(targets, outputs))

	# Recall - How correctly are positives predicted?
	print("Recall (TP / (TP + FN)):", recall_score(targets, outputs))

# Logistic Regression (even though it is a classifier)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)

# MNIST data set of handwritten digits
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

#find the total amount of ink in the top half of the image
def inkSumTop(row):
	return (1000 * sigmoid(row[1:393].sum())) #magic number, but this worked the best

def inkSumBot(row):
	return(1000 * sigmoid(row[393:785].sum())) #magic number, but this worked the best

mnist_train['inksumtop'] = mnist_train.apply(inkSumTop, axis = 1)
mnist_test['inksumtop'] = mnist_train.apply(inkSumTop, axis = 1)

mnist_train['inksumbot'] = mnist_train.apply(inkSumBot, axis = 1)
mnist_test['inksumbot'] = mnist_train.apply(inkSumBot, axis = 1)

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]

# Now let's shuffle the training set to reduce bias opportunities
from sklearn.utils import shuffle
smn_train_targets, smn_train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

# Let's try our Logistic Classifier on the MNIST data, predicting digit 5
from sklearn.linear_model import LogisticRegression



# Softmax Regression or Multinomial Logistic Regression!
print("Training a Multinomial Logistic Regression classifier for ALL digits!")
softmax_reg = LogisticRegression(penalty="none",multi_class="multinomial", solver="saga")
softmax_reg.fit(mnist_train_inputs, mnist_train_targets)
softmax_outputs = softmax_reg.predict(mnist_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))




