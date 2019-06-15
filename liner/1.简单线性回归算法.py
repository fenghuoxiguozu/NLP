import numpy as np
import matplotlib.pyplot as plt
from liner.settings import *


#损失函数,依次传入 x，y
def loss_cost(w,b,length):
    cost=0
    for i in range(length):
        cost +=(y[i]-w*x[i]-b)**2
    return cost/length

#定义均值函数
def average(data):
    ave=0
    length=len(data)
    for i in range(length):
        ave+=data[i]
    return ave/length

#定义拟合函数
def fit(x,y):
    length=len(x)
    ave=average(x)
    x_sum=0     #分母
    y_sum=0     #分子
    b_delta=0   #误差
    for i in range(length):
        y_sum+=y[i]*(x[i]-ave)
        x_sum+=x[i]**2
    w=y_sum/(x_sum-length*(ave**2))

    for i in range(length):
        b_delta+=y[i]-w*x[i]
    b=b_delta/length
    return w,b

def draw(x,y,w,b):
    predict_y=w*x+b
    plt.scatter(x, y)
    plt.plot(x,predict_y,c='r')
    plt.show()

if __name__ == '__main__':
    x=np.random.rand(200)
    noise=np.random.normal(0,0.01,x.shape)
    y=x*0.1+noise

    w,b=fit(x,y)
    loss=loss_cost(w, b, len(x))
    print('loss:',loss)
    draw(x, y, w, b)