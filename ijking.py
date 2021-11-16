import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt
from packages import top_finder

minima, minima_err, maxima, maxima_err = top_finder.find_tops(
    "metingen 15-11-21/ALL0001/F0001CH1.csv", False
)

# F = 1 --> F = 0
transition_1to0 = 4.271676631815181 + 384230.4844685 - 0.3020738
transition_1to0_err = np.sqrt(
    0.00000000000000056 ** 2 + 0.000000062 ** 2 + 0.000000088 ** 2
)

# F = 1 --> F = 2
transition_1to2 = 4.271676631815181 + 384230.4844685 - 0.0729112
transition_1to2_err = np.sqrt(
    0.00000000000000056 ** 2 + 0.000000062 ** 2 + 0.000000032 ** 2
)

delta_f = abs(transition_1to0 - transition_1to2)
delta_f_err = np.sqrt(transition_1to0_err ** 2 + transition_1to2_err ** 2)

delta_t = abs(maxima[0] - maxima[2])
delta_t_err = np.sqrt(maxima_err[0] ** 2 + maxima_err[2] ** 2)

print(f"transition_1to0 = {transition_1to0} +- {transition_1to0_err} GHz")
print(f"transition_1to2 = {transition_1to2} +- {transition_1to2_err} GHz")
print(f"delta_f = {delta_f} +- {delta_f_err} GHz")
print(f"delta_t = {delta_t} +- {delta_t_err} s")

calib = delta_f / delta_t
calib_err = np.sqrt(
    (delta_f_err / delta_t) ** 2 + (delta_f * delta_t_err / (delta_t ** 2)) ** 2
)

print(f"calib = {calib} +- {calib_err} GHz / s")
