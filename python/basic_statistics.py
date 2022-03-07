# exercise 3.2.1
# Computation of mean, SMC, Jaccard, Cosine similarity
from scipy.stats import norm
import numpy as np
from project1 import *
from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show, boxplot

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
# print('Vector:',x)
print('Mean:',mean_x)
print('Standard Deviation:',std_x)
print('Median:',median_x)
print('min, max Range:',min_x, max_x, range_x)

print('Ran Exercise 3.2.1')

########################
#CHECKING NORMAL DISTRIBUTION 
# exercise 4.1.3, 4.2.2
# "2bill_length_mm","3bill_depth_mm","4flipper_length_mm","5body_mass_g"


nb_of_attributes = 4
figure(figsize=(8,7))
u = int(np.floor(np.sqrt(nb_of_attributes)))
v = int(np.ceil(float(nb_of_attributes)/u))


for i in range(1, 5):
    subplot(2,2,i)
    hist(X[:,i], color=(0.2, 0.8-i*0.2, 0.4))
    xlabel(attributeNames[i])

plt.savefig('images/histograms.pdf',bbox_inches = 'tight')
show()


########################
# CHECKING outliers (with box plots ) exe 4.2.3
attribute = 4
figure(figsize=(3, 8))
X = X[:,attribute]
boxplot(X)
xlabel(attributeNames[attribute])
ylabel('g')
plt.savefig('images/bodymass_boxplot.pdf',bbox_inches = 'tight')
show()
plt.close()


