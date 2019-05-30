import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img=plt.imread(r'e1.jpg')
print("压缩前图片shape",img.shape,type(img))  #(480, 320, 3)<class 'numpy.ndarray'>

# plt.imshow(img)
# plt.show()

#降维处理
a=img.reshape((-1,2))
print(a.shape)  #(230400, 2)

kmn=KMeans(n_clusters=16)
y_new=kmn.fit_predict(a)


centers=kmn.cluster_centers_
print(centers)
plt.scatter(a[:,0],a[:,1],c=y_new)
plt.scatter(centers[:,0],centers[:,1],c='r')
plt.show()


# new=np.array(list_img)
# new_img=new.reshape((480, 320,3))
# plt.imshow(new_img)
# plt.show()
# plt.savefig("e2.jpg")
# print("压缩后图片shape",new_img.shape)


