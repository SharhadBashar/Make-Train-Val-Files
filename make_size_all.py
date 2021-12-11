import imageio
import numpy as np
from PIL import Image
import pickle
import json
import os.path
from os import path
import glob
import re
from random import randrange

'''
21 classes:
0:  background

1 to 20 are objects:
1:  Aeroplane
2:  Bicycle
3:  Bird
4:  Boat
5:  Bottle
6:  Bus
7:  Car
8:  Cat
9:  Chair
10: Cow
11: Diningtable
12: Dog
13: Horse
14: Motorbike
15: Person
16: Pottedplant
17: Sheep
18: Sofa
19: Train
20: Tvmonitor

255: doesn't count
'''

class Images:
  def __init__(self, w = 500, h = 281, aug_path = 'SegmentationClassAug'):
    self.size_dict = {
      0: 0,
      1: 0,  2: 0,  3: 0,  4: 0,  5: 0,
      6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
      11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
      16: 0, 17: 0, 18: 0, 19: 0, 20: 0
    }
    self.images = glob.glob(aug_path + '/*.png')

  def check(self, folder):
    if not path.isdir(folder): os.makedirs(folder)

  def open(self, pkl):
    return pickle.load(open(pkl, "rb"))

  def random_num_gen(self, lower = 1, upper = 100):
    return randrange(lower, upper + 1)


class Actual(Images):
  def __init__(self, folder = 'SegmentationClassAugSizeActual'):
    super().__init__()
    self.folder = folder
    self.check(self.folder)
    self.run_actual()

  def run_actual(self):
    for image in self.images:
      size_dict = self.size_dict.copy()
      # Open image and flatten
      im = np.asarray(Image.open(image))
      imFlat = im.flatten()

      # Get size and count
      size = imFlat.shape[0]
      counts = np.unique(imFlat, return_counts = True)
      classPresent = counts[0]
      count = counts[1]

      # Assert lengths of arrays are the same
      assert len(classPresent) == len(count), 'Length of class array not the same as length of count array'

      for i in range(len(classPresent)):
        if classPresent[i] == 255: continue
        size_dict[classPresent[i]] = count[i] / size

      # Write to file
      file = open(self.folder + '/' + image.split('/')[1].split('.')[0] + '.pkl', 'wb')
      pickle.dump(size_dict, file)
      file.close()

class Bucket(Images):
  def __init__(self, folder = 'SegmentationClassAugSizeBucket'):
    super().__init__()
    self.folder = folder
    self.check(self.folder)
    self.run_bucket()

  def run_bucket(self):
    # Open image and flatten
    for image in self.images:
      size_dict = self.size_dict.copy()
      im = np.asarray(Image.open(image))
      imFlat = im.flatten()

      # Get size and count
      size = imFlat.shape[0]
      counts = np.unique(imFlat, return_counts = True)
      classPresent = counts[0]
      count = counts[1]

      # Assert lengths of arrays are the same
      assert len(classPresent) == len(count), 'Length of class array not the same as length of count array'

      # Update dictionary
      for i in range(len(classPresent)):
          if classPresent[i] == 255:
              continue
          approxCount = 0
          approxSize = count[i] / size

          if approxSize <= 0.1: approxCount = 0.1
          elif approxSize > 0.1 and approxSize <= 0.2: approxCount = 0.2
          elif approxSize > 0.2 and approxSize <= 0.3: approxCount = 0.3
          elif approxSize > 0.3 and approxSize <= 0.4: approxCount = 0.4
          elif approxSize > 0.4 and approxSize <= 0.5: approxCount = 0.5
          elif approxSize > 0.5 and approxSize <= 0.6: approxCount = 0.6
          elif approxSize > 0.6 and approxSize <= 0.7: approxCount = 0.7
          elif approxSize > 0.7 and approxSize <= 0.8: approxCount = 0.8
          elif approxSize > 0.8 and approxSize <= 0.9: approxCount = 0.9
          elif approxSize > 0.9 and approxSize <= 1  : approxCount = 1

          size_dict[classPresent[i]] = approxCount

      # Write to file
      file = open(self.folder + '/' + image.split('/')[1].split('.')[0] + '.pkl', 'wb')
      pickle.dump(size_dict, file)
      file.close()

class Unknown(Images):
  def __init__(self, folder = 'SegmentationClassAugSizeUnknown'):
    super().__init__()
    self.folder = folder
    self.check(self.folder)
    self.run_unknown()

  def run_unknown(self):
    # Open image and flatten
    for image in self.images:
      size_dict = self.size_dict.copy()
      size_dict[0] = 0.5
      im = np.asarray(Image.open(image))
      imFlat = im.flatten()

      # Get size and count
      size = imFlat.shape[0]
      counts = np.unique(imFlat, return_counts = True)
      classPresent = counts[0]

      # Remove background and None Class 255
      if (0 in classPresent): classPresent = np.delete(classPresent, np. where(classPresent == 0))
      if (255 in classPresent): classPresent = np.delete(classPresent, np. where(classPresent == 255))

      # Size of each class in each image
      sizeOfClass = 0.5 / len(classPresent)

       # Update dictionary
      for i in range(len(classPresent)):
          size_dict[classPresent[i]] = sizeOfClass

      # Write to file
      file = open(self.folder + '/' + image.split('/')[1].split('.')[0] + '.pkl', 'wb')
      pickle.dump(size_dict, file)
      file.close()

class Shift(Images):
  def __init__(self, percentage = 0, folder = 'SegmentationClassAugSizeShift_', bucket_folder = 'SegmentationClassAugSizeBucket'):
    self.images_pkl = glob.glob(bucket_folder + '/*.pkl')
    super().__init__()
    self.folder = folder + str(percentage)
    self.check(self.folder)
    self.run_shift(percentage)

  def run_shift(self, percentage):
    for image_pkl in self.images_pkl:
      size_dict = self.open(image_pkl).copy()
      for i in range(1, 21):
        if (size_dict[i] != 0):
          object_probability = self.random_num_gen()
          if object_probability <= percentage:
            if (size_dict[i] == 0.1):
              size_dict[i] = 0.2
            elif(size_dict[i] == 0.9):
              size_dict[i] = 0.8
            else:
              bucket_probability = self.random_num_gen(lower = 0, upper = 1)
              if bucket_probability == 0:
                size_dict[i] -= 0.1
              else:
                size_dict[i] += 0.1

      # Write to file
      file = open(self.folder + '/' + image_pkl.split('/')[1], 'wb')
      pickle.dump(size_dict, file)
      file.close()

if __name__ == '__main__':
  Actual()
  Bucket()
  Unknown()
  for i in range(5, 21, 5):
    Shift(i)



















