import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt

fg_freq = 10  # Hz
fg_freq_err = 0.03  # Hz


def find_tops(filename, show_plot=False, log=False):
    """Finds peaks given filename of csv containing data.

    Args:
        filename (string): name of csv file
        show_plot (bool) [optional]: show plot of data with vertical lines where the peaks are. Defaults to False.

    Returns:
        (tuple): tuple containing:
            minima_c (list): list of centres of minima found
            minima_c_err (list): list of errors of minima found
            maxima_c (list): list of centres of maxima found
            maxima_c_err (list): list of errors of maxima found
    """
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
                err = 0.00010
            minima_c_err.append(err)
            if log:
                print(
                    f"[find_tops()] Found minimum at t = {minima_c[-1]} +- {minima_c_err[-1]} s."
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
                err = 0.00010
            maxima_c_err.append(err)

            if log:
                print(
                    f"[find_tops()] Found maximum at t = {maxima_c[-1]} +- {maxima_c_err[-1]} s."
                )
            temp_count = -1
        temp_count += 1

    if show_plot:
        plt.scatter(df["time"], df["volt"], s=2)
        for m in minima_c:
            plt.axvline(m, c="blue")
            plt.text(m, 0.6, str(minima_c.index(m)))
        for m in maxima_c:
            plt.axvline(m, c="red")
            plt.text(m, 0.6, str(maxima_c.index(m)))
        plt.show()

    return (minima_c, minima_c_err, maxima_c, maxima_c_err)


def calibrate(filename):
    """Calibrates ratio of seconds on oscilloscope to frequency (energy) given a dataset to calibrate with the F = 1 --> F = 0 and F = 1 --> F = 2 transitions.

    Args:
        filename (string): name of csv file

    Returns:
        (tuple): tuple containing:
            calib (float): calibration value in GHz/s or MHz/ms
            calib_err (float): calibration value error in GHz/s or MHz/ms"""
    minima, minima_err, maxima, maxima_err = find_tops(filename, True)

    peak1 = int(input("Type the number of the peak that corresponds to F=2 ---> F=3: "))
    peak2 = int(
        input("Type the number of the peak that corresponds to F=2 ---> F=2.5: ")
    )

    # F = 2 --> F = 3
    transition_2to3 = 384230.4844685 - 2.563005979089109 + 0.1937407
    transition_2to3_err = np.sqrt(
        0.000000062 ** 2 + 0.000000000000034 ** 2 + 0.0000046 ** 2
    )

    # F = 2 --> F = 2.5
    transition_2to25 = 384230.4844685 - 2.563005979089109 + (0.1937407 - 0.0729112) / 2
    transition_2to25_err = np.sqrt(
        0.000000062 ** 2
        + 0.000000000000034 ** 2
        + (0.0000090 / 2) ** 2
        + (0.0000032 / 2) ** 2
    )

    delta_f = abs(transition_2to3 - transition_2to25)
    delta_f_err = np.sqrt(transition_2to3_err ** 2 + transition_2to25_err ** 2)

    delta_t = abs(maxima[peak1] - maxima[peak2])
    print(delta_t)
    delta_t_err = np.sqrt(maxima_err[peak1] ** 2 + maxima_err[peak2] ** 2)

    # print(f"transition_1to0 = {transition_1to0} +- {transition_1to0_err} GHz")
    # print(f"transition_1to2 = {transition_1to2} +- {transition_1to2_err} GHz")
    # print(f"delta_f = {delta_f} +- {delta_f_err} GHz")
    # print(f"delta_t = {delta_t} +- {delta_t_err} s")

    calib = -delta_f / delta_t
    calib_err = calib * np.sqrt(
        (delta_t_err / delta_t) ** 2
        + (delta_f_err / delta_f) ** 2
        + (fg_freq_err / fg_freq) ** 2
    )

    print(f"[calibrate()] calibration = {calib} +- {calib_err} GHz / s")
    return (calib, calib_err)
