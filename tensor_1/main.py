# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
# from IPython.display import display, Image

from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import random
from skimage import io


url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'  # Change me to store data elsewhere

#
# def download_progress_hook(count, blockSize, totalSize):
#     """A hook to report the progress of a download. This is mostly intended for users with
#     slow internet connections. Reports every 5% change in download progress.
#     """
#     global last_percent_reported
#     percent = int(count * blockSize * 100 / totalSize)
#
#     if last_percent_reported != percent:
#         if percent % 5 == 0:
#             sys.stdout.write("%s%%" % percent)
#             sys.stdout.flush()
#         else:
#             sys.stdout.write(".")
#             sys.stdout.flush()
#
#         last_percent_reported = percent
#
#
# def maybe_download(filename, expected_bytes, force=False):
#     """Download a file if not present, and make sure it's the right size."""
#     dest_filename = os.path.join(data_root, filename)
#     if force or not os.path.exists(dest_filename):
#         print('Attempting to download:', filename)
#         filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
#         print('\nDownload Complete!')
#     statinfo = os.stat(dest_filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified', dest_filename)
#     else:
#         raise Exception(
#             'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
#     return dest_filename
#
#
# train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
# test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
#
# num_classes = 10
# np.random.seed(133)
#
#
# def maybe_extract(filename, force=False):
#     root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
#     if os.path.isdir(root) and not force:
#         # You may override by setting force=True.
#         print('%s already present - Skipping extraction of %s.' % (root, filename))
#     else:
#         print('Extracting data for %s. This may take a while. Please wait.' % root)
#         tar = tarfile.open(filename)
#         sys.stdout.flush()
#         tar.extractall(data_root)
#         tar.close()
#     data_folders = [
#         os.path.join(root, d) for d in sorted(os.listdir(root))
#         if os.path.isdir(os.path.join(root, d))]
#     if len(data_folders) != num_classes:
#         raise Exception(
#             'Expected %d folders, one per class. Found %d instead.' % (
#                 num_classes, len(data_folders)))
#     print(data_folders)
#     return data_folders
#
#
# train_folders = maybe_extract(train_filename)
# test_folders = maybe_extract(test_filename)
#
# image_size = 28  # Pixel width and height.
# pixel_depth = 255.0  # Number of levels per pixel.
#
#
# def load_letter(folder, min_num_images):
#     """Load the data for a single letter label."""
#     image_files = os.listdir(folder)
#     dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
#                          dtype=np.float32)
#     print(folder)
#     num_images = 0
#     for image in image_files:
#         image_file = os.path.join(folder, image)
#         try:
#             image_data = (imageio.imread(image_file).astype(float) -
#                           pixel_depth / 2) / pixel_depth
#             if image_data.shape != (image_size, image_size):
#                 raise Exception('Unexpected image shape: %s' % str(image_data.shape))
#             dataset[num_images, :, :] = image_data
#             num_images = num_images + 1
#         except (IOError, ValueError) as e:
#             print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
#
#     dataset = dataset[0:num_images, :, :]
#     if num_images < min_num_images:
#         raise Exception('Many fewer images than expected: %d < %d' %
#                         (num_images, min_num_images))
#
#     print('Full dataset tensor:', dataset.shape)
#     print('Mean:', np.mean(dataset))
#     print('Standard deviation:', np.std(dataset))
#     return dataset
#
#
# def maybe_pickle(data_folders, min_num_images_per_class, force=False):
#     dataset_names = []
#     for folder in data_folders:
#         set_filename = folder + '.pickle'
#         dataset_names.append(set_filename)
#         if os.path.exists(set_filename) and not force:
#             # You may override by setting force=True.
#             print('%s already present - Skipping pickling.' % set_filename)
#         else:
#             print('Pickling %s.' % set_filename)
#             dataset = load_letter(folder, min_num_images_per_class)
#             try:
#                 with open(set_filename, 'wb') as f:
#                     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
#             except Exception as e:
#                 print('Unable to save data to', set_filename, ':', e)
#
#     return dataset_names
#
#
# train_datasets = maybe_pickle(train_folders, 45000)
# test_datasets = maybe_pickle(test_folders, 1800)
#
#
# def make_arrays(nb_rows, img_size):
#     if nb_rows:
#         dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
#         labels = np.ndarray(nb_rows, dtype=np.int32)
#     else:
#         dataset, labels = None, None
#     return dataset, labels
#
#
# def merge_datasets(pickle_files, train_size, valid_size=0):
#     num_classes = len(pickle_files)
#     valid_dataset, valid_labels = make_arrays(valid_size, image_size)
#     train_dataset, train_labels = make_arrays(train_size, image_size)
#     vsize_per_class = valid_size // num_classes
#     tsize_per_class = train_size // num_classes
#
#     start_v, start_t = 0, 0
#     end_v, end_t = vsize_per_class, tsize_per_class
#     end_l = vsize_per_class + tsize_per_class
#     for label, pickle_file in enumerate(pickle_files):
#         try:
#             with open(pickle_file, 'rb') as f:
#                 letter_set = pickle.load(f)
#                 # let's shuffle the letters to have random validation and training set
#                 np.random.shuffle(letter_set)
#                 if valid_dataset is not None:
#                     valid_letter = letter_set[:vsize_per_class, :, :]
#                     valid_dataset[start_v:end_v, :, :] = valid_letter
#                     valid_labels[start_v:end_v] = label
#                     start_v += vsize_per_class
#                     end_v += vsize_per_class
#
#                 train_letter = letter_set[vsize_per_class:end_l, :, :]
#                 train_dataset[start_t:end_t, :, :] = train_letter
#                 train_labels[start_t:end_t] = label
#                 start_t += tsize_per_class
#                 end_t += tsize_per_class
#         except Exception as e:
#             print('Unable to process data from', pickle_file, ':', e)
#             raise
#
#     return valid_dataset, valid_labels, train_dataset, train_labels
#
#
# train_size = 200000
# valid_size = 10000
# test_size = 10000
#
# valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
#     train_datasets, train_size, valid_size)
# _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
#
# print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation:', valid_dataset.shape, valid_labels.shape)
# print('Testing:', test_dataset.shape, test_labels.shape)
#
#
# def randomize(dataset, labels):
#     permutation = np.random.permutation(labels.shape[0])
#     shuffled_dataset = dataset[permutation, :, :]
#     shuffled_labels = labels[permutation]
#     return shuffled_dataset, shuffled_labels
#
#
# train_dataset, train_labels = randomize(train_dataset, train_labels)
# test_dataset, test_labels = randomize(test_dataset, test_labels)
# valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
#
# pickle_file = os.path.join(data_root, 'notMNIST.pickle')
#
# try:
#     f = open(pickle_file, 'wb')
#     save = {
#         'train_dataset': train_dataset,
#         'train_labels': train_labels,
#         'valid_dataset': valid_dataset,
#         'valid_labels': valid_labels,
#         'test_dataset': test_dataset,
#         'test_labels': test_labels,
#     }
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#     f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise
#
# statinfo = os.stat(pickle_file)
#print('Compressed pickle size:', statinfo.st_size)


with open('notMNIST.pickle', 'rb') as f:
    entry = pickle.load(f)

for key in entry.keys():
    print (key)
#print(entry["train_labels"][1])
#print(entry["train_dataset"][1].shape)
#a=np.unique(entry["train_labels"])


#plt.imshow(entry["train_dataset"][2000], cmap='gray')
#plt.show()
#digits = datasets.load_digits()
nsamples, nx, ny = entry["train_dataset"].shape

d2_train_dataset = entry["train_dataset"].reshape((nsamples,nx*ny))
idx = np.random.choice(200000, 10000, replace=False)
train_dataset = d2_train_dataset[idx,:]
train_labels = entry["train_labels"][idx]

test_dataset = entry["test_dataset"]
nsamples, nx, ny = test_dataset.shape
d2_test_dataset = test_dataset.reshape((nsamples,nx*ny))
test_labels = entry["test_labels"]
print ("Test Data Size : " , test_dataset.shape)

#digits = datasets.load_digits()
#train_dataset = digits.data
#train_labels = digits.target

#print(entry["train_dataset"][0])
#print("--------")
#print(train_dataset[0])

X = np.asarray(train_dataset)
Y = np.asarray(train_labels)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
#print("######################")
#print(X[0])
X_train = X;
Y_train = Y;
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                    test_size=0.2,
#                                                   random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


#############################################################################
#Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.05
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0,solver = 'lbfgs')
logistic_classifier.fit(X_train, Y_train)

# #############################################################################

# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(rbm.components_):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle('100 components extracted by RBM', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()


img = io.imread('test_14.png', as_grey=True)
# io.imsave('test_3x.png',img)
# img = img.astype(np.float32)
img=img*255
img = (img - 255.0 / 2) / 255.0
#plt.imshow(img, cmap='gray')
nx, ny = img.shape
img_flat = img.reshape(nx * ny)
#print("******", img_flat)
#IMG = np.reshape(img, (1, 784))

res=logistic_classifier.predict(img_flat.reshape(1,-1))
print("truth = ",test_labels[502])
print("guess = ",res)

score = logistic_classifier.score(d2_test_dataset, test_labels)
print("score = " , score)