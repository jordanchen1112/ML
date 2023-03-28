import tensorflow as tf
import numpy as np

# tf.reduce_mean(array,axis=) 
# axis 表示往哪個軸向進行平均, 預設axis = None代表取所有元素之平均值
def loss(y,y_pred):
	return tf.reduce_mean(tf.square(y-y_pred), axis = None)

def predict(x):
	return w * x + b

def train(x,y,epochs = 50, lr =0.0001):
	for e in range(epochs):
		with tf.GradientTape() as g:
			y_pred = predict(x)
			current_loss = loss(y, y_pred)
		dw, db = g.gradient(current_loss, [w, b])

		w.assign_sub(dw * lr)
		b.assign_sub(db * lr)

		print(f'Epoch{e}:Loss:{current_loss.numpy()}')

n = 100
x = np.linspace(0,50,n)
x += np.random.uniform(-10, 10 ,n)
y = np.linspace(0,50,n)
y += np.random.uniform(-10, 10 ,n)

w = tf.Variable(0.0)
b = tf.Variable(0.0)

train(x,y)
print(f'w = {w.numpy()}, b = {b.numpy()}')