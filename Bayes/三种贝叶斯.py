from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


iris=datasets.load_iris()
x=iris.data
y=iris.target
print(y)

#高斯分布，正太分布
gnb1=GaussianNB()
test1=gnb1.fit(x,y).score(x,y)
print(test1)

#伯努利分布
gnb2=BernoulliNB()
test2=gnb2.fit(x,y).score(x,y)
print(test2)

#多项式分布
gnb3=MultinomialNB()
test3=gnb3.fit(x,y).score(x,y)
print(test3)