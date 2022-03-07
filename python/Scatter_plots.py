# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:05:57 2022

@author: Marti
"""

from project1 import *



###### scatter plot of all of them (exe 4.3.2)
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