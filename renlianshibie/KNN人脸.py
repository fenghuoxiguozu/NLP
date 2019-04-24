import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsRegressor


datas=fetch_lfw_people()
data=datas.data  #(13233, 2914)
target=datas.target  #(13233,)
images=datas.images  #(13233, 62, 47)
target_names=datas.target_names  #(5749,)

# plt.imshow(data[0,:].reshape(62,47))   #显示一张人脸
# plt.imshow(images[0])  #显示一张人脸
# plt.imshow(data[2,:1457].reshape(31,47),cmap="gray")  #显示上半人脸

x_train=data[:10000,:1457]
y_trian=data[:10000,1457:]
x_test=data[10000:,:1457]
y_test=data[10000:,1457:]
knn=KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train,y_trian)

y_new=knn.predict(x_test)  #(3233, 1457)

# print(y_new,y_new.shape)
result=np.concatenate((x_test[0],y_new[0]))
plt.imshow(result.reshape((62,47)))
plt.show()