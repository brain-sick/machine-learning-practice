from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
log_reg = LogisticRegression(C=100)
#lower C => log_reg adjusts to majority of data points
#higher C => log_reg adjusts to all data points
log_reg.fit(x_train,y_train)
print('accu on training subset : {:.3f}'.format(log_reg.score(x_train,y_train)))
print('accu on test subset : {:.3f}'.format(log_reg.score(x_test,y_test)))
