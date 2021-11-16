import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt


def find_tops(filename, show_plot):
    df = pd.read_csv(filename)

    minima = []
    maxima = []

    for i in range(len(df["volt"])):
        if i > 100 and i < len(df["volt"]) - 101:
            neighbourhood = list(df["volt"][i - 100 : i + 101])
            neighbourhood_t = list(df["time"][i - 100 : i + 101])
            mid_index = len(neighbourhood) // 2
            mid_val = neighbourhood[mid_index]
            if (
                mid_val == min(neighbourhood)
                and max(neighbourhood[:10]) > mid_val + 0.01600
                and max(neighbourhood[10:]) > mid_val + 0.01600
            ):
                minima.append(neighbourhood_t[len(neighbourhood) // 2])
            elif (
                mid_val == max(neighbourhood)
                and min(neighbourhood[:10]) < mid_val - 0.01600
                and min(neighbourhood[10:]) < mid_val - 0.01600
            ):
                maxima.append(neighbourhood_t[len(neighbourhood) // 2])

    print(minima)
    print(maxima)

    minima_c = []
    maxima_c = []
    minima_c_err = []
    maxima_c_err = []

    temp_count = 0
    for i in range(len(minima)):
        # Check for jumps in time, this indicates a different peak has been reached in the list,
        # or check if this is the end of the loop, in which the last peak needs to be closed.
        if i == len(minima) - 1 or minima[i + 1] - minima[i] > 0.0005:
            # Peak is a collection of points with the same voltage reading
            peak = minima[i - temp_count : i + 1]
            # Centre = average of values of peak
            minima_c.append(sum(peak) / (temp_count + 1))
            # Error = distance centre value to the largest deviation
            err = max(peak) - minima_c[-1]
            if err == 0:
                err = 0.00002
            minima_c_err.append(err)
            print(
                f"[find_tops()] Found minimum at t = {minima_c[-1]} +- {minima_c_err[-1]} ms."
            )
            temp_count = -1
        temp_count += 1

    temp_count = 0
    for i in range(len(maxima)):
        if i == len(maxima) - 1 or maxima[i + 1] - maxima[i] > 0.0005:
            peak = maxima[i - temp_count : i + 1]
            maxima_c.append(sum(peak) / (temp_count + 1))
            err = max(peak) - maxima_c[-1]
            if err == 0:
                err = 0.00002
            maxima_c_err.append(err)
            print(
                f"[find_tops()] Found maximum at t = {maxima_c[-1]} +- {maxima_c_err[-1]} ms."
            )
            temp_count = -1
        temp_count += 1

    if show_plot:
        plt.scatter(df["time"], df["volt"], s=2)
        for m in minima_c:
            plt.axvline(m, c="blue")
        for m in maxima_c:
            plt.axvline(m, c="red")
        plt.show()

    return (minima_c, minima_c_err, maxima_c, maxima_c_err)


print("IJKING:")
find_tops("metingen 15-11-21/ALL0001/F0001CH1.csv", True)

print("METING:")
find_tops("metingen 15-11-21/ALL0002/F0002CH3.csv", True)
