import pandas as pd
from lmfit import models
import numpy as np

df_m = pd.read_csv("", delimiter=",", decimal=".")

df_m["volt_err"] = df_m["volt"] * 0.03

model_m = models.GaussianModel(prefix="g1_") + models.GaussianModel(prefix="g2_")

# fit_m = model_m.fit(data = df_m["volt"], x = df_m["time"], weights = 1/df_m["volt_err"], g1_amplitude = , g1_center = , g1_sigma = , g2_amplitude = , g2_center = , g2_sigma =)
