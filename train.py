"""
Multilayer Percentron Training module
Author: Shibsankar Das
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

# calculate one hot encoding of a given vector
def one_hot_encoding(x):
    n = len(x)
    n_unique = len(np.unique(x))
    one_hot_encode = np.zeros([n,n_unique])
    one_hot_encode[np.arange(n),x] = 1
    return one_hot_encode

# Read and Preprocess training data
traininig_data = pd.read_csv('./mobile-price-classification/train.csv')
traininig_data.fillna(0,inplace=True)
X = traininig_data[traininig_data.columns[0:19]].values
y = traininig_data[traininig_data.columns[20]].values
Y = one_hot_encoding(y)

# Split data-set into Training and Test set
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state=42)

# define and initialize parameters

learning_rate = 0.01
dimension = X.shape[1]
epoch = 100

# define number of neuron for each hidden layer
n_layer_1 = dimension
n_layer_2 = dimension
n_layer_3 = dimension
n_layer_4 = dimension
n_classes = len(np.unique(y))

# define input and output placeholder
x = tf.placeholder(tf.float32,[None,dimension],name='x')
_y = tf.placeholder(tf.float32,[None,n_classes],name='y')

# multilayer perceptron [Layer: 4, Neuron at each Layer {D,D,D,D,# of classes}]
def multilayer_perceptron(X, weights, biases):
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
    return out_layer

# define hidden layer variables
weights = {
    'h1':tf.Variable(tf.truncated_normal([dimension,n_layer_1]), name = 'h1'),
    'h2':tf.Variable(tf.truncated_normal([n_layer_1,n_layer_2]), name = 'h2'),
    'h3':tf.Variable(tf.truncated_normal([n_layer_2,n_layer_3]), name = 'h3'),
    'h4':tf.Variable(tf.truncated_normal([n_layer_3,n_layer_4]), name = 'h4'),
    'out':tf.Variable(tf.truncated_normal([n_layer_4,n_classes]), name = 'h_out')
}

biases = {
    'b1':tf.Variable(tf.truncated_normal([n_layer_1]), name = 'b1'),
    'b2':tf.Variable(tf.truncated_normal([n_layer_2]), name = 'b2'),
    'b3':tf.Variable(tf.truncated_normal([n_layer_3]), name = 'b3'),
    'b4':tf.Variable(tf.truncated_normal([n_layer_4]), name = 'b4'),
    'out':tf.Variable(tf.truncated_normal([n_classes]), name = 'b_out')
}


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
# write graph to log
tf.summary.FileWriter('./graph',sess.graph)

y_output = multilayer_perceptron(x,weights,biases)

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_output,labels=_y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(loss_function)

for i in range(0, epoch):
    sess.run(training_step, feed_dict={x: train_X, _y: train_Y})
    cost = sess.run(loss_function, feed_dict={x: train_X, _y: train_Y})
    accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_output, 1), tf.argmax(_y, 1)), tf.float32)),
                        feed_dict={x: train_X, _y: train_Y})

    print('epoch: ', i, ' , cost: ', cost, ' accuracy: ', accuracy)

saver.save(sess,'./model_1')
sess.close()

print('Training Finished.')





