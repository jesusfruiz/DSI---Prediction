# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import pandas as pd

def get_model(x_train, y_train, random_grid):
    rf = RandomForestRegressor()
    rf_random = GridSearchCV(rf, random_grid, cv = 3, verbose=1, n_jobs = -1)
    rf_random.fit(x_train, y_train)
    return rf_random

vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

andalucia = output['historic'][0]

x_data = {}
y_data = {}

for key in andalucia.keys():
    if key == 'AcumulatedCases':
        array = np.array(andalucia['AcumulatedCases']) + np.array(andalucia['AcumulatedPRC'])
    elif key == 'AcumulatedPRC' or key == 'AcumulatedTestAc' :
        continue
    else:
        array = np.array(andalucia[key])
    if(len(array) == 1): 
        array = array[0]
    x_data[key] = array
#    if key in vars_to_predict:
#        y_data[key] = array
#    else:
#        x_data[key] = array
    
#y['Dates'] = np.array(range(0, len(data['AcumulatedCases'])))
x = pd.DataFrame(x_data)
x = x.drop(columns='label_x')

# Number of trees in random forest
n_estimators = [50, 100, 200, 500, 600]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 30, 60, 80, 100]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)

#regressor = RandomForestRegressor(n_estimators=4, max_depth = 3, criterion='mae', random_state=0)
#regressor.fit(x_train, y_train)
#
#ypred = {}
#ypred['Dates'] = np.array([60, 61, 62, 63, 64, 65, 66])
#ypred = pd.DataFrame(ypred)
#
#pred = regressor.predict(ypred)

init = 55
for var in vars_to_predict:
    regressor = get_model(x.drop(columns=var).loc[:init], x[var].loc[:init], random_grid)
    
    pred = regressor.predict(x.drop(columns=var).loc[init+1:init+7])
    
    print("The prediction and the real results are the following in the var", var)
    print(pred)
    print(x[var].loc[init+1:init+7])
        