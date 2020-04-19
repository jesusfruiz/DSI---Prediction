# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matlab.engine 
eng = matlab.engine.start_matlab() 
output, name_ccaa, iso_ccaa, data_spain = eng.HistoricDataSpain(nargout=4)
