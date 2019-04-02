import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df=pd.read_table('lenses.txt',header=None)

dict0={'young':0,'pre':1,'presbyopic':2,}
df[0]=df[0].map(dict0)   #手动生成数值

def get_num(value):  #自动生成数值
    return np.argwhere(df[i].unique()==value)[0][0]

for i in range(0,5):
    df[i]=df[i].map(get_num)

tree=DecisionTreeRegressor()
# x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,:4],df.iloc[:,4],test_size=0.2)
x_train,y_train=df.iloc[:-2,:4],df.iloc[:-2,4]
x_test,y_test=df.iloc[-2:,:4],df.iloc[-2:,4]

tree.fit(x_train,y_train)
result=tree.predict(x_test)
print(result)