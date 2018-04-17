from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage import data
from PIL import Image
import cv2
import math
from scipy import ndimage


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

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


img = cv2.imread("test23.png", 0)

# resize the images and invert it (black background)
gray = cv2.resize(255 - img, (28, 28))
gray = np.asarray(img)
# while np.sum(gray[0]) == 0:
#   gray = gray[1:]
#
# while np.sum(gray[:, 0]) == 0:
#   gray = np.delete(gray, 0, 1)
#
# while np.sum(gray[-1]) == 0:
#   gray = gray[:-1]
#
# while np.sum(gray[:, -1]) == 0:
#   gray = np.delete(gray, -1, 1)

rows, cols = gray.shape
if rows > cols:
      factor = 25.0/rows
      rows = 25
      cols = int(round(cols*factor))
      gray = cv2.resize(gray, (cols,rows))
else:
      factor = 25.0/cols
      cols = 25
      rows = int(round(rows*factor))
      gray = cv2.resize(gray, (cols, rows))

colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted
io.imsave('test23xx.png',gray)
print (gray)
plt.imshow(gray, cmap='gray')
plt.show()
#img = Image.open('test_5.png')
#gry = img.convert('L')
#grarray = np.asarray(gry)
#print("#################\n",grarray)
#bw = (grarray > grarray.mean())*255
#io.imsave('test_3x.png',img)
# print("#################\n",img)
# img = img.astype(np.float32)
# img = img*255
# print("???????????????????\n",img)
# img = (img - 255.0 / 2) / 255.0
# plt.imshow(img, cmap='gray')
# plt.show()
# io.imsave('test_3x.png',img)
# #print(img)
# print("------------\n" , test_dataset[0])
# nx, ny = img.shape
# img_flat = img.reshape(nx * ny)
# print("**********\n" , img_flat)
