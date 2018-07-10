import numpy as np

def update(X,Y,theta,alpha):
    m=np.shape(X)[0]
    n=np.shape(X)[1]
    for j in range(n):
        b=0
        for i in range(m):
            a=sum(theta*X[i])
            a=a-Y[i]
            b=b+(a*X[i][j])
        theta[j]=theta[j]-(alpha*b)/m
            
X=np.loadtxt(r'C:\Users\RAJ\Desktop\featuresX.txt',delimiter=',',dtype='float')
Y=np.loadtxt(r'C:\Users\RAJ\Desktop\priceY.txt')
X[:,0]=X[:,0]/max(X[:,0])
X[:,1]=X[:,1]/max(X[:,1])
X=np.insert(X,0,1,axis=1)
theta=np.full(3,3)
for i in range(10000):
    update(X,Y,theta,0.01)
print(theta)

