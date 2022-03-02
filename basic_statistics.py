# exercise 3.2.1
# Computation of mean, SMC, Jaccard, Cosine similarity
import numpy as np
from project1 import *

x = np.array(raw_data[:, 6]) #CHANGE COLUMN HERE. IF DISCRETE USE dtype=int
# x = np.array([-0.68, -2.11, 2.39, 0.26, 1.46, 1.33, 1.03, -0.41, -0.33, 0.47])

# Compute values
mean_x = x.mean()
std_x = x.std(ddof=1)   #standard deviation
median_x = np.median(x)
min_x = x.min()
max_x = x.max()
range_x = x.max()-x.min()

# Display results
# For discrete (integer) values, use parameter dtype=int
print('Vector:',x)
print('Mean:',mean_x)
print('Standard Deviation:',std_x)
print('Median:',median_x)
print('min, max Range:',min_x, max_x, range_x)

print('Ran Exercise 3.2.1')