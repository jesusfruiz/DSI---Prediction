# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
import math
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,\
                                 validation_curve
import numpy as np
import pandas as pd

def get_model(x_train, y_train, random_grid):
    rf = RandomForestRegressor()
    rf_random = GridSearchCV(rf, random_grid, cv = 5, verbose=1, n_jobs = -1)
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)
    return rf_random

def draw_validation_curve(train_scores, test_scores, param_range):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
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

vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

andalucia = output['historic'][0]

#data_spain = transform_matlab_data(data_spain)
x_data = {}
y_data = {}

andalucia = transform_matlab_data(andalucia)
andalucia = andalucia.drop(columns='label_x')

n_estimators = [5, 15, 30, 60, 120, 300, 600, 1000]
max_features = [1, 2, 3, 4, 5, 6, 7, 8]
max_features.append("auto")
max_features.append("sqrt")
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features}
#max_features = [1,2,3,4,5,6,7,8]


init = 60
x = andalucia.astype('int32')       
for var in vars_to_predict:
    x_train = x.drop(columns=var).loc[:init]
    y_train = x[var].loc[:init]
    x_test = x.drop(columns=var).loc[init+1:init+7]
    y_test = x[var].loc[init+1:init+7]
    
    regressor = get_model(x_train, y_train, random_grid)
    
    pred = regressor.predict(x_test)
    
    print("The prediction and the real results are the following in the var", var)
    print(pred)
    print(y_test)
        