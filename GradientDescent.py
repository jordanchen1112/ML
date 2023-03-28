import numpy as np
# from numpty import arange
import matplotlib.pyplot as plt
from sympy import * 

def f(x):
	return 2 * (x**2) + x 

def f1(x):
	return 4*x + 1

def GradientDescent(x_start, f1, epochs,lr):
	x_array = np.zeros(epochs+1)
	#一開始權重 = x_start
	x = x_start
	x_array[0] = x
	for i in range(epochs):
		#更新權重
		dx = f1(x)
		x = x - (dx *lr)
		x_array[i+1] = x

	return x_array 

#Hyperparameters
x_start = 10
epochs = 40
lr = 0.45

#作圖f(x)
t = np.arange(-10,10,0.01) # datype(t) = array
plt.plot(t,f(t),c='black')

#作圖權重(x)梯度下降的變化
w = GradientDescent(x_start, f1, epochs,lr)
plt.plot(w,f(w),c='green')
plt.scatter(w,f(w),c='black') #input datype cannot be 'list', need to be 'array'
print(np.round(w,2))
#Show the plot
plt.xlabel('x')
plt.ylabel('Loss')
plt.title('Gradient Descent')
plt.show()

