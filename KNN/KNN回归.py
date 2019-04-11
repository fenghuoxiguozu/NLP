import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

x=np.sort(5*np.random.rand(40,1),axis=0)
y=np.sin(x).ravel()

y[::5]+=1*(0.5-np.random.rand(8))  #破坏数据的整齐性

step=np.linspace(0,5,100)[:,np.newaxis]  #(100,)


knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x,y)
test=knn.predict(step)

plt.scatter(x,y,c="k",label="data")
plt.plot(step,test,c="g",label="predict")
plt.axis("tight")
plt.legend(loc='best')  #设置图中每个图例标签位置
plt.show()