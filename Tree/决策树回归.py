import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

data=200*np.random.randn(100,1)-100
x_train=np.sort(data,axis=0)#100*1

y_train=np.pi*np.array([np.sin(x_train).ravel(),np.cos(x_train).ravel()])#2*100
y_train=y_train.transpose()#x旋转数据  100*2
y_train[::4]+=np.random.randn(25,2)   #插入垃圾数据

#训练数据
tree1=DecisionTreeRegressor(max_depth=5)
tree2=DecisionTreeRegressor(max_depth=10)
tree3=DecisionTreeRegressor(max_depth=50)
tree1.fit(x_train,y_train)
tree2.fit(x_train,y_train)
tree3.fit(x_train,y_train)

#预测数据
x_test=np.arange(-100,100,0.1).reshape((-1,1))
result1=tree1.predict(x_test)
result2=tree2.predict(x_test)
result3=tree3.predict(x_test)

#画图 原图，深度5,10,50
fig=plt.figure(figsize=(12,12))
axes1=fig.add_subplot(221)
s1=axes1.scatter(y_train[:,0],y_train[:,1],label="original")
axes1.legend()

axes2=fig.add_subplot(222)
s2=axes2.scatter(result1[:,0],result1[:,1],label="max_depth=5")
axes2.legend()

axes3=fig.add_subplot(223)
s3=axes3.scatter(result2[:,0],result2[:,1],label="max_depth=10")
axes3.legend()

axes4=fig.add_subplot(224)
s4=axes4.scatter(result3[:,0],result3[:,1],label="max_depth=50")
axes4.legend()

fig.legend((s1,s2,s3,s4),("1","2","3","4"),"upper right")
plt.show()





