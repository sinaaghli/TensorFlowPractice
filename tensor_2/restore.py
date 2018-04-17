from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from skimage import io

pickle_file = 'notMNIST.pickle'

#
# with open(pickle_file, 'rb') as f:
#   save = pickle.load(f)
#   train_dataset = save['train_dataset']
#   train_labels = save['train_labels']
#   valid_dataset = save['valid_dataset']
#   valid_labels = save['valid_labels']
#   test_dataset = save['test_dataset']
#   test_labels = save['test_labels']
#   del save  # hint to help gc free up memory
#   print('Training set', train_dataset.shape, train_labels.shape)
#   print('Validation set', valid_dataset.shape, valid_labels.shape)
#   print('Test set', test_dataset.shape, test_labels.shape)
#
#
#
# image_size = 28
# num_labels = 10
#
# def reformat(dataset, labels):
#   dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
#   # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
#   labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
#   return dataset, labels
# train_dataset, train_labels = reformat(train_dataset, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
# test_dataset, test_labels = reformat(test_dataset, test_labels)
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)





sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
# Now, access the op that you want to run.
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
restored_training = graph.get_tensor_by_name("train_to_restore:0")
img = io.imread('test_13x.png', as_grey=True)
print (img)
img = img*255
img = (img.astype(float) - 256/2) / 256
print("???????????????????\n",img)

nx, ny = img.shape
img_flat = img.reshape(nx * ny)
#print("******" , img_flat)
IMG = np.reshape(img,(1,784))
answer = sess.run(op_to_restore, feed_dict={restored_training: IMG})
print(answer)
# input = tf.placeholder(tf.float32)
# feed_dict = {input: img_flat}
# print("shape of the image mat", test_dataset[678].shape)
# classification = sess.run(op_to_restore, feed_dict)
# print(classification[0])
# # This will print 60 which is calculated
# # using new values of w1 and w2 and saved value of b1.
# plt.imshow(img, cmap='gray')
# plt.show()
