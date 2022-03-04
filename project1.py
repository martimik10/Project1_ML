# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, plot, legend, show,  xlabel, ylabel, xticks, yticks
import os #TODO delete if not needed 
import warnings #suppress plt warnings
# Open it from Spyder in appropriate folder (settings -> working dir -> curr proj dir)
# Check all "DELETE" and "TODO" flags before hand-in
#DELETE From exe 1.5.1 

warnings.filterwarnings("ignore") #ignore ALL warnings (there are about 60 plt warning about update)
    #if struggeling with debugging, turn this off! 

filename = "dataset/penguins.csv"
df = pd.read_csv(filename)

raw_data = df.values

########### LINEAR REGRESSION #################################################
#0:rowid 1:species 2:island 3:bill_length_mm 4:bill_depth_mm	
    # 5:flipper_length_mm	6:body_mass_g	7:sex	8:year
cols = range(3, 7) #for regression we want only the interval attributes

X = raw_data[:, cols] #obtain matrix X
#obtain attribute names 
attributeNames = np.asarray(df.columns[cols]) #['bill_length_mm' 'bill_depth_mm' 'flipper_length_mm' 'body_mass_g']

# -> we want labels to be species
classLabels = raw_data[:, 1]


#obtain class names 
classNames = np.unique(classLabels) # = ['Adelie' 'Chinstrap' 'Gentoo']
classDict = dict(zip(classNames,range(len(classNames)))) #{'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

y = np.array([classDict[cl] for cl in classLabels]) #[0 0 0 2 2 1 1...]

N, M = X.shape
C = len(classNames)
# print(N, M, C)

########################################################################

############## CLASSIFICATION ##########################################
#We want to predict sex on all other meaningful attributes
cols = [1, 3, 4, 5, 6, 7] #we dont care about rowID, year, island
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:, 7] #we want it to be sex
classNames = np.unique(classLabels.astype("str")) 
classDict = dict(zip(classNames,range(len(classNames)))) #{'female': 0, 'male': 1, 'nan': 2}


y = np.array([classDict[cl] for cl in classLabels.astype("str")])
N, M = X.shape
C = len(classNames)

# ##### scatter plot of all of them (exe 4.3.2)
figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
plt.savefig('images/scatter_plot_all.png', dpi=300)
show()
print("best for classification seems to be body mass vs bill depth")
###### Scatter plot of body mass vs bill depth (exe 1.5.4)
i = 4; j = 2;
color = ['r','g', 'b']
plt.title('Penguin classification')
for c in range(len(classNames)):
    idx = y == c
    plt.scatter(x=X[idx, i],
                y=X[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
plt.savefig('images/scatter_plot_2D.png', dpi=300)
plt.show()
