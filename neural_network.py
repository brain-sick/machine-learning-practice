from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer=load_breast_cancer()
print(cancer.data.shape)
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)
'''
mlp =  MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)
print('Acu on Training set : {:.3f}'.format(mlp.score(x_train,y_train)))
print('Acu on Test set : {:.3f}'.format(mlp.score(x_test,y_test)))
'''
mlp=MLPClassifier(max_iter=1000,random_state=42,alpha=1)
scaler=StandardScaler()
x_train_scaled=scaler.fit(x_train).transform(x_train)
x_test_scaled=scaler.fit(x_test).transform(x_test)
mlp.fit(x_train_scaled,y_train)
print('Acu on Training set : {:.3f}'.format(mlp.score(x_train_scaled,y_train)))
print('Acu on Test set : {:.3f}'.format(mlp.score(x_test_scaled,y_test)))

