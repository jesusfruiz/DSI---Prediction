# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

vars_to_predict = ['DailyCases', 'Hospitalized', 'Critical', 'DailyDeaths', 'DailyRecoveries']

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

andalucia = output['historic'][0]

data = {}
y_data = {}

for key in andalucia.keys():
    array = np.array(andalucia[key])
    if(len(array) == 1): 
        array = array[0]
        
    if key in vars_to_predict:
        y_data[key] = array
    else:
        data[key] = array
    
#y['Dates'] = np.array(range(0, len(data['AcumulatedCases'])))
x = pd.DataFrame(data)
y = pd.DataFrame(y_data)
#print(df)
#print(y)
#df = df.drop(columns='label_x')
#    
x = x.drop(columns='label_x')

#regressor = RandomForestRegressor(n_estimators=4, max_depth = 3, criterion='mae', random_state=0)
#regressor.fit(x_train, y_train)
#
#ypred = {}
#ypred['Dates'] = np.array([60, 61, 62, 63, 64, 65, 66])
#ypred = pd.DataFrame(ypred)
#
#pred = regressor.predict(ypred)

for var in vars_to_predict:
    x_train = x.loc[:52]
    y_train = y[var].loc[:52]
    x_test = x.loc[53:]
    y_test = y[var].loc[53:]
    
    regressor = RandomForestRegressor(n_estimators= int(math.sqrt(59)), criterion='mae', random_state=0)
    regressor.fit(x_train, y_train)
    
    pred = regressor.predict(x_test)
    
    print("The prediction and the real results are the following in the var", var)
    print(pred)
    print(y_test)