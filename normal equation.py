import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X=np.loadtxt(r'C:\Users\RAJ\Desktop\featuresX.txt',delimiter=',',dtype='int')
#Y=np.loadtxt(r'C:\Users\RAJ\Desktop\priceY.txt')
Y=np.array([0.5,1,2,0])
#X=np.insert(X,0,1,axis=1)
T=X.transpose()
P=np.matmul(np.matmul((inv(np.matmul(T,X))),T),Y)
#np.save('temp.npy',P)  #save the paramenters array
#P=np.load('temp.npy')
print(P)
'''
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('no of bedrooms')
ax.set_zlabel('cost')
ax.scatter(A,B,Y,c='r',marker='o')
plt.show()
'''
