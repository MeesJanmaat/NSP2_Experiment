import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("")

minima = []
maxima = []

for i in range(len(df["volt"])):
    if i > 4 and i < len(df["volt"]) - 4:
        neighbourhood = [
            df["volt"][i - 3],
            df["volt"][i - 2],
            df["volt"][i - 1],
            df["volt"][i],
            df["volt"][i + 1],
            df["volt"][i + 2],
            df["volt"][i + 3],
        ]
        neighbourhood_t = [
            df["time"][i - 3],
            df["time"][i - 2],
            df["time"][i - 1],
            df["time"][i],
            df["time"][i + 1],
            df["time"][i + 2],
            df["time"][i + 3],
        ]
        if neighbourhood[0] > min(neighbourhood) and neighbourhood[6] > min(
            neighbourhood
        ):
            minima.append(neighbourhood_t[neighbourhood.index(min(neighbourhood))])
        elif neighbourhood[0] < max(neighbourhood) and neighbourhood[6] < max(
            neighbourhood
        ):
            maxima.append(neighbourhood_t[neighbourhood.index(max(neighbourhood))])

print(minima)
print(maxima)
