import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ALL0001/F0001CH1.csv")

minima = []
maxima = []


for i in range(len(df["volt"])):
    if i > 20 and i < len(df["volt"]) - 21:
        neighbourhood = list(df["volt"][i - 20 : i + 21])
        neighbourhood_t = list(df["time"][i - 20 : i + 21])
        mid_val = neighbourhood[len(neighbourhood) // 2]
        if (
            mid_val == min(neighbourhood)
            and neighbourhood[0] > mid_val + 0.00800
            and neighbourhood[-1] > mid_val + 0.00800
        ):
            minima.append(neighbourhood_t[len(neighbourhood) // 2])
        elif (
            mid_val == max(neighbourhood)
            and neighbourhood[0] < mid_val - 0.00800
            and neighbourhood[-1] < mid_val - 0.00800
        ):
            maxima.append(neighbourhood_t[len(neighbourhood) // 2])


print(minima)
print(maxima)