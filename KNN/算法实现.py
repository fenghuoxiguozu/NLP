import sklearn.datasets as datasets
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class MyKNN(object):
    def  __init__(self,K):
        self.K=K

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    #计算测试点到所有训练集的距离
    def distance(self,X_train,b):
        return [np.sqrt((np.sum(a - b)) ** 2) for a in X_train]

    def predict(self,X_test):
        y_predict=[]
        for x in X_test:
            distance=self.distance(X_train,x)
            nearest=np.argsort(distance)     #所有距离排序，返回index
            topK=[self.Y_train[index] for index in nearest[:self.K]]   #返回距离最近K个点的Y
            votes=Counter(topK)   #计数
            y_predict.append(votes.most_common(1)[0][0])   #返回最多值的Y
        return y_predict

    def score(self,Y_test,y_predict):
        acc= accuracy_score(Y_test,y_predict)
        return acc



if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data  # (150, 4)
    y = iris.target  # (150,)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=1, stratify=y)

    knn=MyKNN(K=4)
    knn.fit(X_train,Y_train)
    y_predict=knn.predict(X_test)
    score=knn.score(Y_test,y_predict)
    print(score)

