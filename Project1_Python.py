# Project 1
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, plot, legend, show,  xlabel, ylabel, xticks, yticks
import warnings #suppress plt warnings

# Basic Statistics
from scipy.stats import norm
from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show, boxplot

# Correlation
from scipy.stats import pearsonr

# Scatter plots
# PCA
from scipy.linalg import svd
from mpl_toolkits import mplot3d

def standardize_data(arr):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = arr.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        
        for element in X[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray

#######################################################
#### PROJECT 1
#######################################################
# Open it from Spyder in appropriate folder (settings -> working dir -> curr proj dir)
# Check all "DELETE" and "TODO" flags before hand-in
# DELETE From exe 1.5.1 

warnings.filterwarnings("ignore") #ignore ALL warnings (there are about 60 plt warning about update)
    #if struggeling with debugging, turn this off! 

filename = "dataset/penguins.csv"
df = pd.read_csv(filename)

raw_data = df.values


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



#######################################################
#### BASIC STATISTICS
#######################################################

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


#######################################################
#### CORRELATION
#######################################################
# interval_attributes = range(1, 5)
# for m1 in interval_attributes:
#     for m2 in interval_attributes:
#         # print("pairs", m1, m2, c)
#         x = X[:,m1]
#         y = X[:,m2]
#         corr, _ = pearsonr(x, y)
#         print(attributeNames[m1],":",attributeNames[m2], np.round(corr, 2))
        


#######################################################
#### SCATTER PLOTS
#######################################################
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
# plt.savefig('images/scatter_plot_all.png', dpi=300)
show()
print("best for classification seems to be body mass vs bill depth")
###### Scatter plot of body mass vs bill depth (exe 1.5.4)
i = 4; j = 2;
plt.title('Penguin classification')
for c in range(len(classNames)):
    idx = y == c
    plt.scatter(x=X[idx, i],
                y=X[idx, j], 
                s=50, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
# plt.savefig('images/scatter_plot_2D.png', dpi=300)
plt.show()

#######################################################
#### PCA
#######################################################

########### prepare for analysis of interval attributes
# [2 'Adelie' 'Torgersen' 39.5 17.4 186 3800 'female' 2007]

cols = [3, 4, 5, 6] #we dont care about rowID, year, island, name
X = raw_data[:, cols] #all interval data
X = np.vstack(X[:, :]).astype(np.float) #convert from object to float array in order for SVD to fucking work
attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:, 7] #we want it to be sex
classNames = np.unique(classLabels.astype("str")) 
classDict = dict(zip(classNames,range(len(classNames)))) #{'female': 0, 'male': 1, 'nan': 2}
y = np.array([classDict[cl] for cl in classLabels.astype("str")])
N, M = X.shape
C = len(classNames)

################# preproccessing
# Y = X - np.ones((N,1))*X.mean(axis=0) # no standardization
Y = standardize_data(X) #high standardization

U,S,V = svd(Y,full_matrices=False) # PCA by computing SVD of Y
print(V)


rho = (S*S) / (S*S).sum()  # Compute variance explained by PCA

#what data is percieved by SVD with 3 principal components: 
components = 2
percentage = ((S[:components]*S[:components]).sum())/(S*S).sum()
print("with first", components, " components, we get", round(percentage*100, 2), "%")
#How many we need to surpass 95%?
components = 0
percentage = 0
goal = 95  #ADJUST as needed
while (percentage < goal):
    percentage = (((S[:components+1]*S[:components+1]).sum())/(S*S).sum())*100
    if (components+1 > len(S)):
        print(goal, "% percentage can't be achieved.")
        break
    
    if percentage >= goal:
        print( "To achieve atleast", goal, "%,", components+1, \
          "components were needed. Achieved:", round(percentage, 2), "%" )
        break
        
    components += 1


threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
# plt.savefig('images/variance_yes_standard.pdf',bbox_inches = 'tight')
plt.show()

################ coeffitients ex 2.1.5
# We know that 3 PCA represent 97% of data which is enough.
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
print(attributeNames)
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks([1, 2, 3, 4])
plt.xlabel("Attributes: 1:bill length 2:bill depth 3:flipper length 4:body mass")
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
# plt.savefig('images/PCA_coeffs.pdf',bbox_inches = 'tight')
plt.show()

############# PCA direction scatter 2D
# V=V.T # For the direction of V to fit the convention in the course we transpose
Z = U *S
i = 0 #principal component 1
j = 1 #principal component 2

for c in range(C):
    plt.plot(Z[y==c,i], Z[y==c,j],'.', alpha=.5)
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.title('Zero-mean and unit variance\n' + 'Projection' )
plt.legend(classNames)
plt.axis('equal')
# plt.savefig('images/PCA_projection.pdf',bbox_inches = 'tight')
plt.show()

############# PCA direction scatter 3D
# V=V.T # For the direction of V to fit the convention in the course we transpose
Z = U *S
i = 0 #principal component 1
j = 1 #principal component 2
k = 2 #principal component 3

fig = plt.figure()
ax = plt.axes(projection ='3d')

for c in range(C):
    ax.plot3D(Z[y==c,i], Z[y==c,j], Z[y==c,k],'.', alpha=.5)
ax.set_xlabel('PC'+str(i+1))
ax.set_ylabel('PC'+str(j+1))
ax.set_zlabel('PC'+str(k+1))
ax.set_title('Zero-mean and unit variance\n' + 'Projection' )
ax.legend(classNames)
ax.axis('auto')
plt.savefig('images/PCA_projection_3D.png',dpi = 500)
plt.show()
