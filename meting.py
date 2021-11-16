import pandas as pd
from lmfit import models
import numpy as np
import matplotlib.pyplot as plt
from packages import top_finder

minima, minima_err, maxima, maxima_err = top_finder.find_tops(
    "metingen 15-11-21/ALL0002/F0002CH3.csv", True
)
