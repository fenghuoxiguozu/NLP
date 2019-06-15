import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


digits=datasets.load_digits()
data=digits.data   #(1797, 64)
target=digits.target   #(1797,)
target_names=digits.target_names  #(10,)  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
images=digits.images   #(1797, 8, 8)

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)

item={'KNN回归':KNeighborsRegressor(),
      '逻辑回归':LogisticRegression(),
      '岭回归':Ridge(),
      '决策树':ExtraTreesRegressor()
      }

for key,value in item.items():
    value.fit(x_train,y_train)
    score=value.score(x_test,y_test)
    print(key,"预测精度：",score)

# plt.figure(figsize=(1,1))
# plt.imshow(data[1].reshape((8,8)),cmap="gray")
# plt.show()