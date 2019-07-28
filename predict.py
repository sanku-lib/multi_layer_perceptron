import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

sess = tf.Session()
saver = tf.train.import_meta_graph("{}.meta".format('./model_1'))
model = saver.restore(sess,'./model_1')

predicted_probabilities = tf.nn.softmax(y_output)
predicted_output = tf.argmax(y_output,1)
confidence_score = tf.reduce_max(y_output)

