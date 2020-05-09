# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
import matplotlib.pyplot as plt
import os
from pylab import * # importar todas las funciones de pylab

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import pandas as pd
import random

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

def predict_community_data(ccaa_data, hyperparameters, init):
    for var in vars_to_predict:
        x_train = ccaa_data.drop(columns=var).loc[:init]
        y_train = ccaa_data[var].loc[:init]
        x_test = ccaa_data.drop(columns=var).loc[init+1:init+num_predictions]
        y_test = ccaa_data[var].loc[init+1:init+num_predictions]
        
        regressor = RandomForestRegressor(
                random_state=random.seed(0),
                n_estimators = hyperparameters[var]['n_estimators'],
                max_features = hyperparameters[var]['max_features'])
        
        regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        
        
        print("The prediction and the real results are the following in the var", var)
        pred = list(map(lambda x: round(x), pred))
        print(pred)
        print(y_test)
        file[vars_traductions[var]] += pred
        plot(list(range(init+1, init+num_predictions+1)), pred, list(range(init+1, init+num_predictions+1)), y_test)   # generar el gráfico de la función y=x   
        show()
        
def get_optimal_hyperparameters(data_spain, grid, init, scorer):
    hyperparameters = {}
    data_spain = data_spain.drop(columns='label_x')
    for var in vars_to_predict:
        x_train = data_spain.drop(columns=var).loc[:init]
        y_train = data_spain[var].loc[:init]
    
        rf = RandomForestRegressor(random_state=random.seed(0))
        rf = GridSearchCV(rf, grid, cv = 5, verbose=1, scoring=scorer, n_jobs = 1)
        rf.fit(x_train, y_train)
        hyperparameters[var] = rf.best_params_
    return hyperparameters

vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']
vars_traductions = {
            'DailyCases': 'CASOS', 
            'Hospitalized': 'Hospitalizados', 
            'Critical': 'UCI', 
            'DailyDeaths': 'Fallecidos', 
            'DailyRecoveries': 'Recuperados'
        }

first_day_to_predict = "16-04-2020"
last_day_to_predict = "30-04-2020"
random.seed(0)
np.random.seed(0)
num_predictions = 7

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

n_estimators = [30, 60, 120, 300, 600, 1000]
max_features = [3, 4, 5, 6, 7, 8]
grid = {'n_estimators': n_estimators,
        'max_features': max_features}

scorer = make_scorer(score_function, greater_is_better=False)

data_spain = transform_matlab_data(data_spain)
first_day_index = data_spain.index[data_spain.label_x == first_day_to_predict].tolist()[0]
last_day_index = data_spain.index[data_spain.label_x == last_day_to_predict].tolist()[0]

hyperparameters = get_optimal_hyperparameters(data_spain, grid, first_day_index, scorer)

if not os.path.exists("files"):
    os.mkdir("files")

for day_to_predict in range(first_day_index, last_day_index+1):
    file = {'CCAA': [],
        'FECHA': [],
        'CASOS': [],
        'Hospitalizados': [],
        'UCI': [],
        'Fallecidos': [],
        'Recuperados': []
        }
    for index, ccaa_data in enumerate(output['historic']):
        ccaa_data = transform_matlab_data(ccaa_data)
    
        file['CCAA'] += [iso_ccaa[index]] * num_predictions
        for i in range(day_to_predict+1, day_to_predict+1+num_predictions):    
            file['FECHA'] += [ccaa_data.label_x[i]]
    
        ccaa_data = ccaa_data.drop(columns='label_x')
        print("Calculate prediction for ", name_ccaa[index])
        predict_community_data(ccaa_data, hyperparameters, day_to_predict)
   
    df = pd.DataFrame(file)
    filename = "JFBR_JAGL_"
    filename += data_spain.label_x[day_to_predict].replace("-", "_")
    filename += ".csv"
    df.to_csv("files/" + filename, index=False)    

        