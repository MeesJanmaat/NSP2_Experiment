import pandas as pd
from lmfit import models
import numpy as np

model = (
    models.LinearModel()
    + models.GaussianModel(prefix="gm_")
    + models.GaussianModel(prefix="g1_")
    + models.GaussianModel(prefix="g2_")
    + models.GaussianModel(prefix="g3_")
)

df = pd.read_csv("ALL0000/F0000CH1.csv", delimiter=",", decimal=".")

df["volt_err"] = df["volt"] * 0.03

fit = model.fit(
    data=df["volt"],
    x=df["time"],
    weights=1 / df["volt_err"],
    slope=10,
    intercept=0.5,
    gm_amplitude=-0.4,
    gm_center=0.027,
    gm_sigma=0.001,
    g1_amplitude=0.04,
    g1_center=0.0263,
    g1_sigma=0.0001,
    g2_amplitude=0.3,
    g2_center=0.027,
    g2_sigma=0.0001,
    g3_amplitude=0.3,
    g3_center=0.0275,
    g3_sigma=0.0001,
)

print(
    fit.params["gm_center"].value,
    fit.params["g1_center"].value,
    fit.params["g2_center"].value,
    fit.params["g3_center"].value,
)
print(
    fit.params["gm_amplitude"].value,
    fit.params["g1_amplitude"].value,
    fit.params["g2_amplitude"].value,
    fit.params["g3_amplitude"].value,
)
print(
    fit.params["gm_sigma"].value,
    fit.params["g1_sigma"].value,
    fit.params["g2_sigma"].value,
    fit.params["g3_sigma"].value,
)
print(fit.params["slope"].value, fit.params["intercept"].value)

fit.plot_fit()

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

delta_t = abs(fit.params["g1_center"].value - fit.params["g3_center"].value)
delta_t_err = np.sqrt(
    fit.params["g1_center"].stderr ** 2 + fit.params["g3_center"].stderr ** 2
)

print(f"transition_1to0 = {transition_1to0} +- {transition_1to0_err} GHz")
print(f"transition_1to2 = {transition_1to2} +- {transition_1to2_err} GHz")
print(f"delta_f = {delta_f} +- {delta_f_err} GHz")
print(f"delta_t = {delta_t} +- {delta_t_err} s")

calib = delta_f / delta_t
calib_err = np.sqrt(
    (delta_f_err / delta_t) ** 2 + (delta_f * delta_t_err / (delta_t ** 2)) ** 2
)

print(f"calib = {calib} +- {calib_err} GHz / s")
