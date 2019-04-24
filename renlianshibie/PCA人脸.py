import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  #调节参数
from sklearn.model_selection import train_test_split

datas=fetch_lfw_people(min_faces_per_person=70,
                      resize=1,
                      slice_=(slice(0,250,None),slice(0,250,None)))
x=datas.data  #(1288, 62500)
y=datas.target  #(1288,)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
svc=SVC()
# svc.fit(x_train,y_train)
# svc.score(x_test,y_test)  #0.41

pca=PCA(n_components=150, #主成分
        svd_solver="randomized",  #打乱
        whiten=True)  #白化
pca.fit(x)

x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)

param_grid={"C":[0.2,0.5,0.8,1,3,5,7,9],
            "gamma":[0.001,0.003,0.005,0.008,0.01,0.05,0.08,0,1,0.5,0.8,1]}
gsvc=GridSearchCV(svc,param_grid=param_grid)  #0.8

gsvc.fit(x_train_pca,y_train)
acc=gsvc.score(x_test_pca,y_test)  #0.66
print(acc)