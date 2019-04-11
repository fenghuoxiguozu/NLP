from sklearn.neighbors import KNeighborsClassifier

# 样本测试数据   身高，体重，鞋码
x_train=[[180,75,44],[160,49,39],[155,45,36],[176,60,43],[188,82,45],[168,51,40]]
y_train=['男','女','女','男','男','女']

#创建KNN分类器
knn=KNeighborsClassifier(n_neighbors=3)
#训练数据
knn.fit(x_train,y_train)

#随机数据测试
test_data=[[182,75,42],[170,46,40]]
result=knn.predict(test_data)
print(result)

