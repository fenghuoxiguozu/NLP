import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.datasets as datasets

iris=datasets.load_iris()
x=iris.data
y=iris.target

index=np.arange(150)
np.random.shuffle(index)  #打乱随机数
print(index)

x_train=x[index[:-20]]
y_train=y[index[:-20]]
x_test=x[index[-20:]]
y_test=y[index[-20:]]

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
acc1=tree.score(x_test,y_test)
print("决策树精度:",acc1)

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
acc2=knn.score(x_test,y_test)
print("KNN精度:",acc2)

log=LogisticRegression()
log.fit(x_train,y_train)
acc3=log.score(x_test,y_test)
print("逻辑回归精度:",acc3)
