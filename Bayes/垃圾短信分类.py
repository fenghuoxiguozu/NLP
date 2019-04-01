import pandas as pd
df=pd.read_csv('duanxin.csv')

from sklearn.feature_extraction.text import TfidfVectorizer #文本处理工具
vec=TfidfVectorizer()
train=vec.fit(df['Text']).transform(df['Text'])

from sklearn.naive_bayes import MultinomialNB   #多项式贝叶斯
mnb=MultinomialNB()
mnb.fit(train,df['Label'])   #训练数据
accurate=mnb.score(train,df['Label'])   #训练精度
print(accurate)

x_test=vec.transform(['money','mobile'])
result=mnb.predict(x_test)
print(result)
