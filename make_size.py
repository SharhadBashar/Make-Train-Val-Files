import imageio
import numpy as np
from PIL import Image
import pickle
import json
import os.path
from os import path
import glob
import re

'''
20 classes: 1 to 20
0 is background
255 doesn't count
'''

class Count():
    def __init__(self, imPath):
        self.classCountActual = {
            0: 0,
            1: 0,  2: 0,  3: 0,  4: 0,  5: 0, 
            6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
            11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
            16: 0, 17: 0, 18: 0, 19: 0, 20: 0
        }
        self.classCountApprox = {
            0: 0,
            1: 0,  2: 0,  3: 0,  4: 0,  5: 0, 
            6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
            11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
            16: 0, 17: 0, 18: 0, 19: 0, 20: 0
        }
        self.classCountUnknown = {
            0: 0.5,
            1: 0,  2: 0,  3: 0,  4: 0,  5: 0, 
            6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
            11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
            16: 0, 17: 0, 18: 0, 19: 0, 20: 0
        }
        self.imPath = imPath
        self.outputPath = imPath.split('/')[1].split('.')[0]
        self.check()
        #self.runActual()
        #self.runApprox()
        self.runUnknown()

    def check(self):
        if not path.isdir('SegmentationClassAugSizeActual'): os.makedirs('SegmentationClassAugSizeActual')
        if not path.isdir('SegmentationClassAugSizeApprox'): os.makedirs('SegmentationClassAugSizeApprox')
        if not path.isdir('SegmentationClassAugSizeUnknown'): os.makedirs('SegmentationClassAugSizeUnknown')

    def runActual(self):
        # Open image and flatten
        im = np.asarray(Image.open(self.imPath))
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
            self.classCountActual[classPresent[i]] = count[i] / size
            
        # Write to file
        file = open('SegmentationClassAugSizeActual/' + self.outputPath + '.pkl', 'wb')
        pickle.dump(self.classCountActual, file)
        file.close()
        
        # Open
        # print(pickle.load(open('Count/' + outputPath + '.pkl', "rb")))
    def runApprox(self):
        # Open image and flatten
        im = np.asarray(Image.open(self.imPath))
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


            self.classCountApprox[classPresent[i]] = approxCount
            
        # Write to file
        file = open('SegmentationClassAugSizeApprox/' + self.outputPath + '.pkl', 'wb')
        pickle.dump(self.classCountApprox, file)
        file.close()

    def runUnknown(self):
        # Open image and flatten
        im = np.asarray(Image.open(self.imPath))
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
            self.classCountUnknown[classPresent[i]] = sizeOfClass

        # Write to file
        file = open('SegmentationClassAugSizeUnknown/' + self.outputPath + '.pkl', 'wb')
        pickle.dump(self.classCountUnknown, file)
        file.close()
        
        # Open
        # print(pickle.load(open('Count/' + outputPath + '.pkl', "rb")))

filesPng = glob.glob('SegmentationClassAug/*.png')
for i in range(len(filesPng)):
    _ = Count(filesPng[i])