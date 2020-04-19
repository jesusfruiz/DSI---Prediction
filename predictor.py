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
for key in andalucia.keys():
    print(key)
    data[key] = np.array(andalucia[key])[0]

df = pd.DataFrame(data)
print(df)
    
#data = andalucia['AcumulatedCases']
#data = np.asarray(data)
#data = data[0].reshape(-1, 1)
#
#days = np.asarray(range(len(data[0])))
#regressor = RandomForestRegressor(n_estimators=4, max_depth = 3, criterion='mae', random_state=0)
#regressor.fit(data[0], days)
#
#pred = regressor.predict(list(range(len(days), len(days)+7)))