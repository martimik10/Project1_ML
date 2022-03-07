# -*- coding: utf-8 -*-


from project1 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

interval_attributes = range(1, 5)
for m1 in interval_attributes:
    for m2 in interval_attributes:
        # print("pairs", m1, m2, c)
        x = X[:,m1]
        y = X[:,m2]
        corr, _ = pearsonr(x, y)
        print(attributeNames[m1],":",attributeNames[m2], np.round(corr, 2))