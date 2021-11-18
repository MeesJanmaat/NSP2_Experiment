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

S = 1 / 2
L = 1
I = 3 / 2
J = S + L
F = I + J

g_F3 = (1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))) * (
    (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
)
print(f"g_F3 = {g_F3}")

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

    delta_t = abs(minima[valley_index] - maxima[peak_index])
    delta_t_err = np.sqrt(minima_err[valley_index] ** 2 + maxima_err[peak_index] ** 2)

    delta_E = calib * delta_t * h * 1e9
    delta_E_err = delta_E * np.sqrt(
        (calib_err / calib) ** 2
        + (delta_t_err / delta_t) ** 2
        + (fg_freq_err / fg_freq) ** 2
    )

    print(
        f"[get_energy()] delta_E = {delta_E} +- {delta_E_err} J ({delta_E / h * 1e-9} MHz) (file: {filename})"
    )
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

current_df = pd.read_csv("stroommetingen.csv", delimiter=";")

print(current_df)

energies = []
energies_err = []

# manual
# for file in files:
#    energy, energy_err = get_energy(file)
#    energies.append(energy)
#    energies_err.append(energy_err)

# auto
valleys = [6, 2, 4, 3, 1, 0, 0, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1]
peaks = [3, 3, 3, 1, 1, 2, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0]
for i in range(len(files)):
    energy, energy_err = get_energy(files[i], valleys[i], peaks[i])
    energies.append(energy)
    energies_err.append(energy_err)

energies = np.array(energies)
energies_err = np.array(energies_err)


def try_fit(m_Fl, m_Fr):
    print(f"[try_fit()] FITTING WITH VALUES m_F = {m_Fl}, m_F = {m_Fr}")

    const = ((m_Fl - m_Fr) * g_F2 * mu_B * mu_0 / length) - (
        (m_Fl + 1 - m_Fr + 1) * g_F3 * mu_B * mu_0 / length
    )

    const_err = const * np.sqrt(
        (mu_0_err / mu_0) ** 2 + (mu_B_err / mu_B) ** 2 + (length_err / length) ** 2
    )

    lin_vals = energies / const
    lin_vals_err = lin_vals * np.sqrt(
        (energies_err / energies) ** 2 + (const_err / const) ** 2
    )

    lin_func = lambda I, N: I * N

    model = models.Model(lin_func)

    fit = model.fit(
        data=lin_vals,
        I=current_df["current"],
        weights=1 / lin_vals_err,
        N=700,
    )
    print(fit.fit_report())
    # fit.plot_fit()
    # plt.show()

    return fit.params["N"].value


N_values = {}
for i in range(-2, 3):
    for j in range(-2, 3):
        N_values[(i, j)] = try_fit(i, j)

print(N_values)
plt.hist(N_values.values(), bins=25)
plt.show()
