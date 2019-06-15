import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

df=pd.read_csv(r'C:\Users\Administrator\Desktop\NLP\liner\iris.csv',header=None)

# x,y=np.split(df.values,(4,),axis=1)  #<class 'numpy.ndarray'>
x=df[[0,1,2,3]]   #<class 'pandas.core.frame.DataFrame'>
y=pd.Categorical(df[4]).codes  #类型转变编号 #<class 'numpy.ndarray'>

#取值
x=x[[0,1]].values
y=y.ravel()

lr=Pipeline([("sc",StandardScaler()),
             ("poly",PolynomialFeatures()),
             ("clf",LogisticRegression())])
lr.fit(x,y)


x_min=x[:,0].min()-0.5
x_max=x[:,0].max()+0.5
y_min=x[:,1].min()-0.5
y_max=x[:,1].max()+0.5
t1=np.linspace(x_min,x_max,500)
t2=np.linspace(y_min,y_max,500)
xx,yy=np.meshgrid(t1,t2)

x_test=np.stack((xx.ravel(),yy.ravel()),axis=1)
y_new=lr.predict(x_test)
y_new=y_new.reshape(xx.shape)

cmap1=ListedColormap(['#FFAAAA','#D2C5BA','#AAFFAA'])
cmap2=ListedColormap(['#5C2DE3','#22F43C','#3F542B'])
plt.pcolormesh(xx,yy,y_new,cmap=cmap1)
plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap2)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('逻辑回归与绘图')
plt.legend()
plt.show()

