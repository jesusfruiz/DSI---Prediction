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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import pandas as pd
import statistics

def get_model(x_train, y_train, grid, score):
    rf = RandomForestRegressor()
    rf = GridSearchCV(rf, grid, cv = 5, verbose=1, scoring=score, n_jobs = -1)
    rf.fit(x_train, y_train)
    print(rf.best_params_)
    print(rf.scorer_)
    print(rf.best_estimator_)
    print("The best score is", rf.best_score_)
    return rf

def score_function(y_pred, y_true):
    diff = np.abs(y_true - y_pred, dtype=np.float64)
    result = np.divide(diff, y_true, out=np.zeros_like(diff), where=y_true!=0, dtype=np.float64)
    ret = np.mean(result)
    return ret

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

vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

n_estimators = [30, 60, 120, 300, 600, 1200]
max_features = [1, 2, 3, 4, 5, 6, 7, 8]
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
    

        