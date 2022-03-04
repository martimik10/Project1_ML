# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:09:41 2022

@author: Marti
"""
# ex2.1.3
from project1 import *
from scipy.linalg import svd

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

# Subtract mean value from data
# Y = X - np.ones((N,1))*X.mean(axis=0)
Y = standardize_data(X)
# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

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

print('Ran Exercise 2.1.3')


