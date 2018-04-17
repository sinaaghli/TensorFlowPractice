# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import matplotlib.pyplot as plt
from skimage import io


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

plt.imshow(test_dataset[233], cmap='gray')
#io.imsave('test_6.png',test_dataset[233])
#plt.show()
#print(np.amax(test_dataset[233]))
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# # With gradient descent training, even this much data is prohibitive.
# # Subset the training data for faster turnaround.
# train_subset = 10000
#
# graph = tf.Graph()
# with graph.as_default():
#     # Input data.
#     # Load the training, validation and test data into constants that are
#     # attached to the graph.
#     tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
#     tf_train_labels = tf.constant(train_labels[:train_subset])
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#
#     # Variables.
#     # These are the parameters that we are going to be training. The weight
#     # matrix will be initialized using random values following a (truncated)
#     # normal distribution. The biases get initialized to zero.
#     weights = tf.Variable(
#         tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#     # Training computation.
#     # We multiply the inputs with the weight matrix, and add biases. We compute
#     # the softmax and cross-entropy (it's one operation in TensorFlow, because
#     # it's very common, and it can be optimized). We take the average of this
#     # cross-entropy across all training examples: that's our loss.
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#
#     # Optimizer.
#     # We are going to find the minimum of this loss using gradient descent.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     # These are not part of training, but merely here so that we can report
#     # accuracy figures as we train.
#     train_prediction = tf.nn.softmax(logits,name="op_to_restore")
#     valid_prediction = tf.nn.softmax(
#         tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
# num_steps = 801
#
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
#
# with tf.Session(graph=graph) as session:
#   # This is a one-time operation which ensures the parameters get initialized as
#   # we described in the graph: random weights for the matrix, zeros for the
#   # biases.
#   tf.global_variables_initializer().run()
#   #saver = tf.train.Saver()
#   # Create a builder
#   #builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
#   print('Initialized')
#   for step in range(num_steps):
#     # Run the computations. We tell .run() that we want to run the optimizer,
#     # and get the loss value and the training predictions returned as numpy
#     # arrays.
#     _, l, predictions = session.run([optimizer, loss, train_prediction])
#     if (step % 100 == 0):
#       print('Loss at step %d: %f' % (step, l))
#       print('Training accuracy: %.1f%%' % accuracy(
#         predictions, train_labels[:train_subset, :]))
#       # Calling .eval() on valid_prediction is basically like calling run(), but
#       # just to get that one numpy array. Note that it recomputes all its graph
#       # dependencies.
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
#   input = tf.placeholder(tf.float32)
#   feed_dict = {input : test_dataset[678]}
#   pred = session.run(train_prediction, feed_dict)
#   print(pred)
#
#


  #builder.add_meta_graph_and_variables(session,
  #                                     [tf.saved_model.tag_constants.TRAINING],
  #                                     signature_def_map=None,
  #                                     assets_collection=None)
  #builder.save()
  #saver.save(session, '/Users/nimaaghli/PycharmProjects/tensor_2/my_test_model.ckpt')






#
# batch_size = 100
#
# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32,
#                                       shape=(None, image_size * image_size)) #this is x
#     tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels)) # this is y_
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     weights = tf.Variable(
#         tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits) #this is y
#     print(train_prediction.shape)
#     valid_prediction = tf.nn.softmax(
#         tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
# num_steps = 18001
#
# with tf.Session(graph=graph) as session:
#   tf.global_variables_initializer().run()
#   print("Initialized")
#   for step in range(num_steps):
#     # Pick an offset within the training data, which has been randomized.
#     # Note: we could use better randomization across epochs.
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     # Generate a minibatch.
#     batch_data = train_dataset[offset:(offset + batch_size), :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     # Prepare a dictionary telling the session where to feed the minibatch.
#     # The key of the dictionary is the placeholder node of the graph to be fed,
#     # and the value is the numpy array to feed to it.
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 500 == 0):
#       print("Minibatch loss at step %d: %f" % (step, l))
#       print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#       print("Validation accuracy: %.1f%%" % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
#
  # img = io.imread('test_4.png', as_grey=True)
  # #io.imsave('test_3x.png',img)
  # #img = img.astype(np.float32)
  # #img=img*255
  # #img = (img - 255.0 / 2) / 255.0
  # plt.imshow(img, cmap='gray')
  # #plt.show()
  # #print(img)
  # #print("---------" , test_dataset[0])
  # nx, ny = img.shape
  # img_flat = img.reshape(nx * ny)
  # print("******" , img_flat)
  # IMG = np.reshape(img,(1,784))
  # answer = session.run(train_prediction, feed_dict={tf_train_dataset: IMG})
  # print(answer)

#
#



#1 hidden layer

# num_nodes = 1024
# batch_size = 150
#
# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size),name="train_to_restore")
#     tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
#     biases_1 = tf.Variable(tf.zeros([num_nodes]))
#     weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
#     biases_2 = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
#     relu_layer = tf.nn.relu(logits_1)
#     logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=tf_train_labels))
#
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#     #optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
#
#     # Predictions for the training
#     train_prediction = tf.nn.softmax(logits_2,name="op_to_restore")
#
#     # Predictions for validation
#     logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
#     relu_layer = tf.nn.relu(logits_1)
#     logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
#
#     valid_prediction = tf.nn.softmax(logits_2)
#
#     # Predictions for test
#     logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
#     relu_layer = tf.nn.relu(logits_1)
#     logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
#
#     test_prediction = tf.nn.softmax(logits_2)
#
# num_steps = 9001
#
# with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     #saver = tf.train.Saver()
#     #builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
#     print("Initialized")
#     for step in range(num_steps):
#         # Pick an offset within the training data, which has been randomized.
#         # Note: we could use better randomization across epochs.
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         # Generate a minibatch.
#         batch_data = train_dataset[offset:(offset + batch_size), :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it.
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#         _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 500 == 0):
#             print("Minibatch loss at step {}: {}".format(step, l))
#             print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
#             print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
#     print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))
#     # builder.add_meta_graph_and_variables(session,
#     #                                      [tf.saved_model.tag_constants.TRAINING],
#     #                                      signature_def_map=None,
#     #                                      assets_collection=None)
#     # #builder.save()
#     #saver.save(session, '/Users/nimaaghli/PycharmProjects/tensor_2/my_test_model.ckpt')
#


num_nodes_1 = 2048
num_nodes_2 = 1024
batch_size = 170

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size),name="train_to_restore")
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes_1]))
    biases_1 = tf.Variable(tf.zeros([num_nodes_1]))

    weights_2 = tf.Variable(tf.truncated_normal([num_nodes_1, num_nodes_2]))
    biases_2 = tf.Variable(tf.zeros([num_nodes_2]))

    weights_3 = tf.Variable(tf.truncated_normal([num_nodes_2, num_labels]))
    biases_3 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)

    logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
    relu_layer_2 = tf.nn.relu(logits_2)

    logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_3, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_3,name="op_to_restore")

    # Predictions for validation
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
    relu_layer_2 = tf.nn.relu(logits_2)
    logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
    valid_prediction = tf.nn.softmax(logits_3)

    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
    relu_layer_2 = tf.nn.relu(logits_2)
    logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
    test_prediction = tf.nn.softmax(logits_3)

num_steps = 19001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    #saver = tf.train.Saver()
    #builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
    print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))
    # builder.add_meta_graph_and_variables(session,
    #                                      [tf.saved_model.tag_constants.TRAINING],
    #                                      signature_def_map=None,
    #                                      assets_collection=None)
    # #builder.save()
    #saver.save(session, '/Users/nimaaghli/PycharmProjects/tensor_2/my_test_model.ckpt')


