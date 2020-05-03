# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
import matplotlib.pyplot as plt
from pylab import * # importar todas las funciones de pylab

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import pandas as pd
import random

def get_model(x_train, y_train, grid, score):
    rf = RandomForestRegressor(random_state=random.seed(0))
    rf = GridSearchCV(rf, grid, cv = 5, verbose=1, scoring=score, n_jobs = 1)
    rf.fit(x_train, y_train)
    return rf

def score_function(y_pred, y_true):
    diff = np.abs(y_true - y_pred, dtype=np.float64)
    result = np.divide(diff, y_true, out=np.zeros_like(diff), where=y_true!=0, dtype=np.float64)
    ret = np.mean(result)
    return ret
    
def transform_matlab_data(matlab_data):
    data = {}
    
    for key in matlab_data.keys():
        if key == 'AcumulatedCases':
            array = np.array(matlab_data['AcumulatedCases']) + \
                                    np.array(matlab_data['AcumulatedPRC'])
        elif key == 'AcumulatedPRC' or key == 'AcumulatedTestAc' :
            continue
        else:
            array = np.array(matlab_data[key])
                
        if(len(array) == 1): 
            array = array[0]
                    
        data[key] = array
        
    return pd.DataFrame(data)

def predict_community_data(ccaa_data, grid, init, score):
    for var in vars_to_predict:
        x_train = ccaa_data.drop(columns=var).loc[:init]
        y_train = ccaa_data[var].loc[:init]
        x_test = ccaa_data.drop(columns=var).loc[init+1:init+7]
        y_test = ccaa_data[var].loc[init+1:init+7]
        
        regressor = get_model(x_train, y_train, grid, score)
        
        pred = regressor.predict(x_test)
        
        print("The prediction and the real results are the following in the var", var)
        print(pred)
        print(y_test)
        plot([61,62,63,64,65,66,67], pred, [61,62,63,64,65,66,67], y_test)   # generar el gráfico de la función y=x   
        show()

vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']
random.seed(0)
np.random.seed(0)

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

n_estimators = [5, 30, 60, 120, 300, 600]
max_features = [3, 4, 5, 6, 7, 8]
max_features.append("auto")
max_features.append("sqrt")
grid = {'n_estimators': n_estimators,
        'max_features': max_features}
score = make_scorer(score_function, greater_is_better=False)

init = 60
for index, ccaa_data in enumerate(output['historic']):
    ccaa_data = transform_matlab_data(ccaa_data)
    ccaa_data = ccaa_data.drop(columns='label_x')
    print("Calculate prediction for ", name_ccaa[index])
    predict_community_data(ccaa_data, grid, init, score)
    

        