# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)

andalucia = output['historic'][0]

data = {}
y = {}
for key in andalucia.keys():
    array = np.array(andalucia[key])
    if(len(array) == 1): 
        array = array[0]
    data[key] = array
    
y['Dates'] = np.array(range(0, len(data['AcumulatedCases'])))
df = pd.DataFrame(data)
y = pd.DataFrame(y)
print(df)
print(y)
df = df.drop(columns='label_x')
    
regressor = RandomForestRegressor(n_estimators=4, max_depth = 3, criterion='mae', random_state=0)
regressor.fit(y, df)

ypred = {}
ypred['Dates'] = np.array([59, 60, 61, 62, 63, 64, 65])
ypred = pd.DataFrame(ypred)

pred = regressor.predict(ypred)