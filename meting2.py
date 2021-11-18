import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt
from packages import nsp2exp

calib, calib_err = nsp2exp.calibrate("ALL0001/F0001CH1.csv")

# constants
h = 6.62607015e-34

length = 0.072  # m
length_err = 0.001  # m
impedance = 0.036  # H
frontal_area = 22.295e-4  # m^2
frontal_area_err = 0.205e-4  # m^2

S = 1 / 2
L = 0
I = 3 / 2
J = S + L
F = I + J

g_F2 = (1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))) * (
    (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
)
print(f"g_F2 = {g_F2}")
mu_0 = 1.25663706212e-6
mu_0_err = 0.00000000019e-6
mu_B = 9.274009994e-24
mu_B_err = 0.000000057e-24

fg_freq = 10  # Hz
fg_freq_err = 0.03  # Hz

def get_energy(filename, valley_index=-1, peak_index=-1):
    if valley_index == -1 and peak_index == -1:
        minima, minima_err, maxima, maxima_err = nsp2exp.find_tops(filename, True)
        valley_index = int(
            input("[get_energy()] Type the number of the valley of interest: ")
        )
        peak_index = int(
            input("[get_energy()] Type the number of the peak of interest: ")
        )
    else:
        minima, minima_err, maxima, maxima_err = nsp2exp.find_tops(filename, False)



def try_fit(m_Fl, m_Fr):
    