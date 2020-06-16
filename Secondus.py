__author__ = 'Kripa Dharan'

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import tensorflow as tf
from tensorflow import keras

mvpStats = pd.read_csv("mvp_votings.csv",index_col=False,header=None)
mvpStats.info()

mvpStats = mvpStats.drop([0], axis = 1)
mvpStats.columns = mvpStats.loc[0]
mvpStats = mvpStats.drop(0)

mvpStats['next_share'] = 0

for i in range(1, len(mvpStats) + 1):
	player = mvpStats['player'][i]
	print(player)
	j = i + 1
	found = False
	while j < len(mvpStats) and found == False:
		if mvpStats['player'][j] == player:
			next_share = mvpStats['award_share'][j]
			found = True
		j += 1
	if found == False:
		next_share = 0
	mvpStats['next_share'][i] = next_share

enc = OrdinalEncoder()
mvpStats[['season', 'player']] = enc.fit_transform(mvpStats[['season', 'player']])

mvpStats = mvpStats.drop(['player'], axis = 1)
mvpStats = mvpStats[mvpStats.columns].astype(float)

targets = mvpStats['next_share']
inputs = mvpStats.drop(['next_share'], axis = 1)

train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size = 0.15, random_state = 0)



"""
lin_reg = LinearRegression()
lin_reg.fit(train_inputs, train_targets) 
outputs = lin_reg.predict(inputs)
mse = mean_squared_error(targets, outputs)
rsq = r2_score(targets, outputs)
print(mse)
print(rsq)
"""
"""
tree = DecisionTreeRegressor(max_depth = 5)
tree.fit(train_inputs, train_targets)
print(tree.score(train_inputs, train_targets))


tree.predict(test_inputs)
print(tree.score(test_inputs, test_targets))
"""


neigh = KNeighborsRegressor(n_neighbors = 10)
neigh.fit(train_inputs, train_targets)
print(neigh.score(train_inputs, train_targets))

neigh.predict(test_inputs)
print(neigh.score(test_inputs, test_targets))


this_year = mvpStats[624:]
this_year_targets = this_year['next_share']
this_year_inputs = this_year.drop(['next_share'], axis = 1)

print(neigh.predict(this_year_inputs))



