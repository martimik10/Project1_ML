# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:09:41 2022

@author: Marti
"""
# ex2.1.3
from project1 import *
from scipy.linalg import svd
import matplotlib.pyplot as plt
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
