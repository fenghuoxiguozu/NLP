from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets as datasets
# 分割数据的模块，把数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

#自带蝴蝶样本 'data': array([[。。。],。。。[...]]  'target':array([0, 0, 0,... 2,2,2]
iris=datasets.load_iris()

#提取样本数据
x_data=iris.data[::2]
y_data=iris.target[::2]

# 将数据集分割成 训练集 与 测试集，切顺序是打乱的。其中测试集占20%
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)

result=knn.predict((x_test[:4]))
print(result)
#测试评分
# print(knn.score(x_test,y_test))


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #绘图引用模块

cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])
#绘制散点图
plt.scatter(iris.data[:,2],iris.data[:,3],c=iris.target,cmap=cmap)
plt.plot(x_train,y_train)
plt.show()

