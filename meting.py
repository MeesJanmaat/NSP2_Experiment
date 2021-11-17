import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt
from packages import nsp2exp

calib, calib_err = nsp2exp.calibrate("ALL0001/F0001CH1.csv")

function_gen = 10  # Hz
function_gen_err = 0.03  # Hz


def get_energy(filename):
    """Calculates energy difference between peaks for F = 1 with m_F = -1 and m_F = 1 using a given dataset

    Args:
        filename (string): name of csv file

    Returns:
        (tuple): tuple containing:
            delta_E (float): value of energy difference in MHz
            delta_E_err (float): error of energy difference in MHz"""
    minima, minima_err, maxima, maxima_err = nsp2exp.find_tops(filename, True)

    valley_index = int(
        input("[get_energy()] Type the number of the valley of interest: ")
    )
    peak_index = int(input("[get_energy()] Type the number of the peak of interest: "))

    delta_t = abs(minima[valley_index] - maxima[peak_index])
    delta_t_err = np.sqrt(minima_err[valley_index] ** 2 + maxima_err[peak_index] ** 2)

    delta_E = calib * delta_t
    delta_E_err = np.sqrt((delta_t * calib_err) ** 2 + (calib * delta_t_err) ** 2)

    print(f"[get_energy()] delta_E = {delta_E} +- {delta_E_err} MHz (file: {filename})")
    return (delta_E, delta_E_err)


files = [
    "metingen 15-11-21/ALL0001/F0001CH3.csv",
    "metingen 15-11-21/ALL0002/F0002CH3.csv",
    "metingen 15-11-21/ALL0003/F0003CH3.csv",
    "metingen 15-11-21/ALL0004/F0004CH3.csv",
    "metingen 15-11-21/ALL0005/F0005CH3.csv",
    "metingen 15-11-21/ALL0006/F0006CH3.csv",
    "metingen 15-11-21/ALL0007/F0007CH3.csv",
    "metingen 15-11-21/ALL0008/F0008CH3.csv",
    "metingen 15-11-21/ALL0009/F0009CH3.csv",
    "metingen 15-11-21/ALL0010/F0010CH3.csv",
    "metingen 15-11-21/ALL0011/F0011CH3.csv",
    "metingen 15-11-21/ALL0012/F0012CH3.csv",
    "metingen 15-11-21/ALL0013/F0013CH3.csv",
    "metingen 15-11-21/ALL0014/F0014CH3.csv",
    "metingen 15-11-21/ALL0015/F0015CH3.csv",
    "metingen 15-11-21/ALL0016/F0016CH3.csv",
    "metingen 15-11-21/ALL0017/F0017CH3.csv",
]

energies = []
energies_err = []

for file in files:
    E, E_err = get_energy(file)
    energies.append(E)
    energies_err.append(E_err)
