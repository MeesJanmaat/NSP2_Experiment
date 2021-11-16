import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt
from packages import nsp2exp

minima, minima_err, maxima, maxima_err = nsp2exp.find_tops(
    "metingen 15-11-21/ALL0002/F0002CH3.csv", True
)

calib, calib_err = nsp2exp.calibrate("metingen 15-11-21/ALL0001/F0001CH1.csv")

valley_index = int(input("Type the number of the valley of interest: "))
peak_index = int(input("Type the number of the peak of interest: "))

delta_t = abs(minima[valley_index] - maxima[peak_index])
delta_t_err = np.sqrt(minima_err[valley_index] ** 2 + maxima_err[peak_index] ** 2)

delta_E = calib * delta_t
delta_E_err = np.sqrt((delta_t * calib_err) ** 2 + (calib * delta_t_err) ** 2)

print(f"delta_E = {delta_E} +- {delta_E_err} MHz")
