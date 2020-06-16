
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Linear Regression with scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from itertools import combinations

lin_reg = LinearRegression()

# Explore the Boston Housing data set
boston = pd.read_csv('boston_housing.csv')

"""
print(boston.info())
print(boston.head())

# Description of Boston Housing data set:
# CRIME RATE =  per capita crime rate by town
# LARGE LOT = proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUSTRY = proportion of non-retail business acres per town
# RIVER = Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX = nitric oxides concentration (parts per 10 million)
# ROOMS = average number of rooms per dwelling
# PRIOR 1940 = proportion of owner-occupied units built prior to 1940
# EMP DISTANCE = weighted distances to five Boston employment centres
# HWY ACCESS = index of accessibility to radial highways
# PROP TAX RATE = full-value property-tax rate per $10,000
# STU TEACH RATIO = pupil-teacher ratio by town
# AFR AMER = 1000(AFA - 0.63)^2 where AFA is the proportion of African Americans by town
# LOW STATUS = % lower status of the population
# MEDIAN VALUE = Median value of owner-occupied homes in $1000’s

# Creator: Harrison, D. and Rubinfeld, D.L.
# This is a copy of UCI ML housing dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

boston.hist(figsize=(14,7))
plt.title("Histogram")
plt.show()
boston.boxplot(figsize=(14,7))
plt.title("Box plot")
plt.show()

corr_matrix=boston.corr()
print(corr_matrix["MEDIAN VALUE"].sort_values(ascending=False))

pd.plotting.scatter_matrix(boston[ ['MEDIAN VALUE','LOW STATUS','ROOMS','INDUSTRY','NOX','PROP TAX RATE','STU TEACH RATIO'] ], figsize=(14,7))
plt.title("Scatter Matrix")
plt.show()

plt.scatter(boston['LOW STATUS'], boston['MEDIAN VALUE'])
plt.title("Scatter of Median Value(y) vs Low Status(x)")
plt.show()
"""

# Setup a sample regression, using scikit
boston_inputs = boston[ ['LOW STATUS'] ] # You can add more columns to this list...
boston_targets = boston['MEDIAN VALUE']

# Train the weights
lin_reg.fit(boston_inputs,boston_targets)

# Generate outputs / Make Predictions
boston_outputs = lin_reg.predict(boston_inputs)

# What's our error?
boston_mse = mean_squared_error(boston_targets, boston_outputs)
# What's our R^2? (amount of output variance explained by these inputs)
boston_r2 = r2_score(boston_targets, boston_outputs)

print("MSE using LOW STATUS (scikit way): " + str(boston_mse*len(boston)))
print("R^2 using LOW STATUS (scikit way): " + str(boston_r2))
print("Weights/Coefficients of Regression: " + str(lin_reg.coef_))
"""
plt.scatter(boston_inputs, boston_targets)
plt.plot(boston_inputs, boston_outputs, c="orange")
plt.xlabel("% lower status of the population")
plt.ylabel("Median value of homes in $1000's")
plt.title('Regression of % lower status vs median home value')
plt.show()

residuals = boston_targets-boston_outputs
plt.scatter(boston_inputs, residuals, c="red")
plt.title('Residuals from regression of % lower status vs median home value')
plt.show()
"""
"""
# Linear Regression the numpy way, for comparison:

inputs = boston.as_matrix(columns=['LOW STATUS'])
inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
targets = boston.as_matrix(columns=['MEDIAN VALUE'])

weights = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)
outputs = np.dot(inputs,weights)
error = np.sum((targets-outputs)**2)

print("MSE using LOW STATUS (numpy way): " +str(error))
"""
def getInputsandTargets(inputs, targets):
	return (boston[inputs], boston[targets])

def trainWeights(inputs, targets):
	lin_reg.fit(inputs, targets) 

def makePrediction(inputs):
	return lin_reg.predict(inputs)

def meanSquaredError(outputs, targets):
	return mean_squared_error(targets, outputs)

def rsquared(outputs, targets):
	return r2_score(targets, outputs)

def runRegression(inputColumn, targetColumn):
	boston_inputs, boston_targets = getInputsandTargets(inputColumn, targetColumn)
	trainWeights(boston_inputs, boston_targets)
	boston_outputs = makePrediction(boston_inputs)
	mse = meanSquaredError(boston_outputs, boston_targets)
	r2 = rsquared(boston_outputs, boston_targets)
	print("MSE using " + str(inputColumn) + " (scikit way): " + str(mse*len(boston)))
	print("R^2 using " + str(inputColumn) + " (scikit way): " + str(r2))
	print("Weights/Coefficients of Regression: " + str(lin_reg.coef_))
	return (mse*len(boston))
"""
#1
runRegression(['LOW STATUS'], 'MEDIAN VALUE')
#2
runRegression(['ROOMS'], 'MEDIAN VALUE')
#3
runRegression(['LOW STATUS', 'ROOMS'], 'MEDIAN VALUE')
#4
boston['ROOMS^2'] = boston['ROOMS']**2
runRegression(['ROOMS', 'ROOMS^2'], 'MEDIAN VALUE')
#5
boston['LOW STATUS^2'] = boston['LOW STATUS']**2
runRegression(['LOW STATUS', 'LOW STATUS^2'], 'MEDIAN VALUE')
#6
runRegression(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2'], 'MEDIAN VALUE')
#7
boston['LOWROOMS'] = boston['LOW STATUS'] * boston['ROOMS']
runRegression(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2', 'LOWROOMS'], 'MEDIAN VALUE')
#8
sigchanges = {}
for col in boston.columns:
	origerror = 10637
	if not col in ['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2', 'LOWROOMS', 'MEDIAN VALUE']:
		print('Regression for #6 with ' + str(col))
		error = runRegression(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2', col], 'MEDIAN VALUE')
		percentchangeerror = (error - origerror)/origerror
		if percentchangeerror < -0.03:
			sigchanges[col] = error
print(sigchanges)
"""
#9
bestCombo = []
bestError = runRegression(['ROOMS'], 'MEDIAN VALUE')
values = []
for col in boston.columns:
	if not col == 'MEDIAN VALUE':
		values.append(col)
for i in range(1, len(values) + 1):
	combine = combinations(values, i)
	for combo in combine:
		combo = list(combo)
		error = runRegression(combo, 'MEDIAN VALUE')
		if error<bestError:
			bestError = error
			bestCombo = combo
print(bestCombo)
print(bestError)


