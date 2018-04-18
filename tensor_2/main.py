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
  save = pickle.load(f, encoding='latin1')
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
#
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
#
# ##Fully connect with 3 hidden layers with dropout 96.4 and learning rate decey
# num_nodes_1 = 2048
# num_nodes_2 = 1024
# num_nodes_3 = 300
# batch_size = 200
#
# hidden_layer_1_keep_prob = 0.5
# hidden_layer_2_keep_prob = 0.7
# hidden_layer_3_keep_prob = 0.8
# beta_1 = 1e-5
# beta_2 = 1e-5
# beta_3 = 1e-5
# beta_4 = 1e-5
#
# hidden_layer_1_stddev = np.sqrt(3.0/784)
# hidden_layer_2_stddev = np.sqrt(2.0/num_nodes_1)
# hidden_layer_3_stddev = np.sqrt(1.0/num_nodes_2)
# output_layer_stddev = np.sqrt(2.0/num_nodes_3)
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
#     weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes_1],stddev=hidden_layer_1_stddev))
#     biases_1 = tf.Variable(tf.zeros([num_nodes_1]))
#
#     weights_2 = tf.Variable(tf.truncated_normal([num_nodes_1, num_nodes_2], stddev=hidden_layer_2_stddev))
#     biases_2 = tf.Variable(tf.zeros([num_nodes_2]))
#
#     weights_3 = tf.Variable(tf.truncated_normal([num_nodes_2, num_nodes_3], stddev=hidden_layer_3_stddev))
#     biases_3 = tf.Variable(tf.zeros([num_nodes_3]))
#
#     weights_4 = tf.Variable(tf.truncated_normal([num_nodes_3, num_labels], stddev=output_layer_stddev))
#     biases_4 = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
#     relu_layer = tf.nn.dropout(tf.nn.relu(logits_1),hidden_layer_1_keep_prob)
#
#     logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
#     relu_layer_2 = tf.nn.dropout(tf.nn.relu(logits_2),hidden_layer_2_keep_prob)
#
#     logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
#     relu_layer_3 = tf.nn.dropout(tf.nn.relu(logits_3),hidden_layer_3_keep_prob)
#
#     out = tf.matmul(relu_layer_3, weights_4) + biases_4
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=tf_train_labels))
#     #loss += (beta_1 * tf.nn.l2_loss(weights_1) +
#     #        beta_2 * tf.nn.l2_loss(weights_2) +
#     #        beta_3 * tf.nn.l2_loss(weights_3) +
#     #        beta_4 * tf.nn.l2_loss(weights_4))
#
#     # Learn with exponential rate decay.
#     global_step = tf.Variable(0, trainable=False)
#     starter_learning_rate = 0.4
#     learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
#
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step);
#     #optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
#
#     # Predictions for the training
#     train_prediction = tf.nn.softmax(out, name="op_to_restore")
#
#     # Predictions for validation
#
#     # Setup validation prediction step.
#     validation_hidden_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
#     validation_hidden_layer_2 = tf.nn.relu(
#         tf.matmul(validation_hidden_layer_1, weights_2) + biases_2)
#     validation_hidden_layer_3 = tf.nn.relu(
#         tf.matmul(validation_hidden_layer_2, weights_3) + biases_3)
#     validation_logits = tf.matmul(validation_hidden_layer_3, weights_4) + biases_4
#     valid_prediction = tf.nn.softmax(validation_logits)
#
#
#
#     # Training computation.
#     test_hidden_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
#     test_hidden_layer_2 = tf.nn.relu(tf.matmul(test_hidden_layer_1, weights_2) + biases_2)
#     test_hidden_layer_3 = tf.nn.relu(tf.matmul(test_hidden_layer_2, weights_3) + biases_3)
#     test_logits = tf.matmul(test_hidden_layer_3, weights_4) + biases_4
#     test_prediction = tf.nn.softmax(test_logits)
#
# num_steps = 20000
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


##Fully connect with 3 hidden layers with dropout 96.4 and learning rate decey
num_nodes_1 = 3048
num_nodes_2 = 2048
num_nodes_3 = 1024
num_nodes_4 = 500
batch_size = 180

hidden_layer_1_keep_prob = 0.5
hidden_layer_2_keep_prob = 0.6
hidden_layer_3_keep_prob = 0.7
hidden_layer_4_keep_prob = 0.8
beta_1 = 1e-5
beta_2 = 1e-5
beta_3 = 1e-5
beta_4 = 1e-5

hidden_layer_1_stddev = np.sqrt(3.0/784)
hidden_layer_2_stddev = np.sqrt(3.0/num_nodes_1)
hidden_layer_3_stddev = np.sqrt(3.0/num_nodes_2)
hidden_layer_4_stddev = np.sqrt(3.0/num_nodes_3)
output_layer_stddev = np.sqrt(2.0/num_nodes_4)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size),name="train_to_restore")
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes_1],stddev=hidden_layer_1_stddev))
    biases_1 = tf.Variable(tf.zeros([num_nodes_1]))

    weights_2 = tf.Variable(tf.truncated_normal([num_nodes_1, num_nodes_2], stddev=hidden_layer_2_stddev))
    biases_2 = tf.Variable(tf.zeros([num_nodes_2]))

    weights_3 = tf.Variable(tf.truncated_normal([num_nodes_2, num_nodes_3], stddev=hidden_layer_3_stddev))
    biases_3 = tf.Variable(tf.zeros([num_nodes_3]))

    weights_4 = tf.Variable(tf.truncated_normal([num_nodes_3, num_nodes_4], stddev=hidden_layer_4_stddev))
    biases_4 = tf.Variable(tf.zeros([num_nodes_4]))

    weights_5 = tf.Variable(tf.truncated_normal([num_nodes_4, num_labels], stddev=output_layer_stddev))
    biases_5 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relu_layer = tf.nn.dropout(tf.nn.relu(logits_1),hidden_layer_1_keep_prob)

    logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
    relu_layer_2 = tf.nn.dropout(tf.nn.relu(logits_2),hidden_layer_2_keep_prob)

    logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
    relu_layer_3 = tf.nn.dropout(tf.nn.relu(logits_3),hidden_layer_3_keep_prob)

    logits_4 = tf.matmul(relu_layer_3, weights_4) + biases_4
    relu_layer_4 = tf.nn.dropout(tf.nn.relu(logits_4), hidden_layer_4_keep_prob)

    out = tf.matmul(relu_layer_4, weights_5) + biases_5
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=tf_train_labels))
    # loss += (beta_1 * tf.nn.l2_loss(weights_1) +
    #        beta_2 * tf.nn.l2_loss(weights_2) +
    #        beta_3 * tf.nn.l2_loss(weights_3) +
    #        beta_4 * tf.nn.l2_loss(weights_4)+
    #        beta_4 * tf.nn.l2_loss(weights_4))
    # Learn with exponential rate decay.
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.3
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step);
    #optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # Predictions for the training
    train_prediction = tf.nn.softmax(out, name="op_to_restore")

    # Predictions for validation

    # Setup validation prediction step.
    validation_hidden_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    validation_hidden_layer_2 = tf.nn.relu(
        tf.matmul(validation_hidden_layer_1, weights_2) + biases_2)
    validation_hidden_layer_3 = tf.nn.relu(
        tf.matmul(validation_hidden_layer_2, weights_3) + biases_3)
    validation_hidden_layer_4 = tf.nn.relu(
        tf.matmul(validation_hidden_layer_3, weights_4) + biases_4)
    validation_logits = tf.matmul(validation_hidden_layer_4, weights_5) + biases_5
    valid_prediction = tf.nn.softmax(validation_logits)



    # Training computation.
    test_hidden_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
    test_hidden_layer_2 = tf.nn.relu(tf.matmul(test_hidden_layer_1, weights_2) + biases_2)
    test_hidden_layer_3 = tf.nn.relu(tf.matmul(test_hidden_layer_2, weights_3) + biases_3)
    test_hidden_layer_4 = tf.nn.relu(tf.matmul(test_hidden_layer_3, weights_4) + biases_4)
    test_logits = tf.matmul(test_hidden_layer_4, weights_5) + biases_5
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 20000

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

