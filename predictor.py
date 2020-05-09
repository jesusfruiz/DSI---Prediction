# -*- coding: utf-8 -*-
"""
Práctica de predicción de las asignatura de Desarrollo de Sistemas Inteligentes
del Máster oficial de Ingeniería informática de la UCLM
Autores:
    Jesús Fernández-Bermejo Ruiz
    Jorge Alberto Gómez León
"""

import matlab.engine 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pylab import * # importar todas las funciones de pylab
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import random


def score_function(y_pred, y_true):
    # Calculate the difference between the observed values and predicted values
    diff = np.abs(y_true - y_pred, dtype=np.float64)
    # Divide diff between observed values. If observed is 0 the division is 0
    result = np.divide(diff, y_true, out=np.zeros_like(diff), where=y_true!=0, dtype=np.float64)
    # Calculate the mean of the error in the 7 days
    ret = np.mean(result)
    return ret
    
def transform_matlab_data(matlab_data):
    data = {}
    
    for key in matlab_data.keys():
        # Because a change in the original data we must combine AcumulatedCases
        # and AcumulatedPRC
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


def predict_community_data(ccaa_data, hyperparameters, prediction_day_index):
    for var in vars_to_predict:
        # x_train and y_train contains the data until the day of prediction
        # x_test and y_test contains the data for the next seven days
        x_train = ccaa_data.drop(columns=var).loc[:prediction_day_index]
        y_train = ccaa_data[var].loc[:prediction_day_index]
        x_test = ccaa_data.drop(columns=var).loc[prediction_day_index+1:prediction_day_index+num_predictions]
        y_test = ccaa_data[var].loc[prediction_day_index+1:prediction_day_index+num_predictions]
        
        regressor = RandomForestRegressor(
                random_state=random.seed(0),
                n_estimators = hyperparameters[var]['n_estimators'],
                max_features = hyperparameters[var]['max_features'])
        
        regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
                
        print("The prediction and the real results are the following in the var", var)
        # round data to the near integer
        pred = list(map(lambda x: round(x), pred))
        print(pred)
        print(y_test)
        # Save the data in the right column
        file[vars_traductions[var]] += pred
        plot(list(range(prediction_day_index+1, prediction_day_index+num_predictions+1)),\
             pred, list(range(prediction_day_index+1, prediction_day_index+num_predictions+1)),\
             y_test)   # generar el gráfico de la función y=x   
        show()

        
def get_optimal_hyperparameters(data_spain, grid, prediction_day_index, scorer):
    hyperparameters = {}
    data_spain = data_spain.drop(columns='label_x')
    # We make a grid search to find the hyperparameters the result in minimal error
    # Because we make fold os 5 units, the search use the 80% of the fata to train
    # and the 20% to test
    for var in vars_to_predict:
        x_train = data_spain.drop(columns=var).loc[:prediction_day_index]
        y_train = data_spain[var].loc[:prediction_day_index]
    
        rf = RandomForestRegressor(random_state=random.seed(0))
        rf = GridSearchCV(rf, grid, cv = 5, verbose=1, scoring=scorer, n_jobs = 1)
        rf.fit(x_train, y_train)
        hyperparameters[var] = rf.best_params_
    return hyperparameters

def save_in_csv(file):
    df = pd.DataFrame(file)
    filename = "JFBR_JAGL_"
    filename += data_spain.label_x[day_to_predict].replace("-", "_")
    filename += ".csv"
    df.to_csv("files/" + filename, index=False) 
    

####### MAIN #######
    
## Global Variables ##
vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']
vars_traductions = {
            'DailyCases': 'CASOS', 
            'Hospitalized': 'Hospitalizados', 
            'Critical': 'UCI', 
            'DailyDeaths': 'Fallecidos', 
            'DailyRecoveries': 'Recuperados'
        }

first_day_to_predict = "15-04-2020"
last_day_to_predict = "30-04-2020"
# We predict the next 7 days
num_predictions = 7
# Values to optimize hyperparametes
n_estimators = [30, 60, 120, 300, 600, 1000]
max_features = [3, 4, 5, 6, 7, 8]
grid = {'n_estimators': n_estimators,
        'max_features': max_features}
# Generate seed to get always the same results
print(type(random))
random.seed(0)
np.random.seed(0)

# Execute the matlab program to get data
eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

# We make a scorer to optimize hyperparameters (ERRORABSOLUTO/VALOROBSERVAD0)*100.
# greater_is_better=False because the function returns the error of the prediction
scorer = make_scorer(score_function, greater_is_better=False)

data_spain = transform_matlab_data(data_spain)
# Get the index for the first day and last day to predict
first_day_index = data_spain.index[data_spain.label_x == first_day_to_predict].tolist()[0]
last_day_index = data_spain.index[data_spain.label_x == last_day_to_predict].tolist()[0]

# Optimize hyperparameters with data from Spain in 15-04-2020
hyperparameters = get_optimal_hyperparameters(data_spain, grid, first_day_index, scorer)

if not os.path.exists("files"):
    os.mkdir("files")

for day_to_predict in range(first_day_index, last_day_index+1):
    # Variable to build a dataframe and save the data into a csv file
    file = {'CCAA': [],
        'FECHA': [],
        'CASOS': [],
        'Hospitalizados': [],
        'UCI': [],
        'Fallecidos': [],
        'Recuperados': []
        }
    # Predict data for all the communities in one day
    for index, ccaa_data in enumerate(output['historic']):
        ccaa_data = transform_matlab_data(ccaa_data)
        
        # Add CCAA and FECHA fields for the next seven days in the file
        file['CCAA'] += [iso_ccaa[index]] * num_predictions
        for i in range(day_to_predict+1, day_to_predict+1+num_predictions):    
            file['FECHA'] += [ccaa_data.label_x[i]]
        
        #Remove label_x columm which is useless for prediction
        ccaa_data = ccaa_data.drop(columns='label_x')
        
        print("Calculate prediction for ", name_ccaa[index])
        predict_community_data(ccaa_data, hyperparameters, day_to_predict)
    
    save_in_csv(file)   

        