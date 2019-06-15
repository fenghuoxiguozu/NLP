import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


diabetes=datasets.load_diabetes()
x=diabetes.data  #(442, 10)
x=x[:,np.newaxis,2]  # (442, 1)
y=diabetes.target  #(442,)
y=y.reshape(442,1)
# print(x,y)

x_train=x[:-10]
x_test=x[-10:]
y_train=y[:-10]
y_test=y[-10:]

lrg=LogisticRegression()
lrg.fit(x_train,y_train)
y_new=lrg.predict(x_test)

plt.scatter(x_test,y_test)
plt.plot(x_test,y_new,"r")
plt.show()


