import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets as datasets
from matplotlib.colors import ListedColormap

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

K=15 #计算周围15个点
h=0.2  #图片x,y每一步步长
cmap1=ListedColormap(['#FFAAAA','#D2C5BA','#AAFFAA'])
cmap2=ListedColormap(['#5C2DE3','#22F43C','#B4ED2B'])

knn=KNeighborsClassifier(n_neighbors=K)
knn.fit(x,y)

#描述图片显示范围
x_min,x_max=x[:,0].min()-1,x[:,0].max()-1
y_min,y_max=x[:,1].min()-1,x[:,1].max()-1
print(x_min,x_max,y_min,y_max)

#生成网格
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

#预测
print("z:",np.c_[xx.ravel(),yy.ravel()])
z=knn.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

#显示背景颜色
plt.pcolormesh(xx,yy,z,cmap=cmap1)

#显示点颜色
plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap2)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("分类")
plt.show()