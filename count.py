import imageio
import numpy as np
from PIL import Image
import pickle
import json
import os.path
from os import path
import glob

'''
20 classes: 1 to 20
0 is background
255 doesn't count
'''

class Count():
    def __init__(self, imPath):
        self.classCount = {
            0:0,
            1: 0,  2: 0,  3: 0,  4: 0,  5: 0, 
            6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
            11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
            16: 0, 17: 0, 18: 0, 19: 0, 20: 0
        }
        self.imPath = imPath
        self.check()
        self.run()

    def check(self):
        if not path.isdir('cls_png_size_actual'): os.makedirs('cls_png_size_actual')

    def run(self):
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
            self.classCount[classPresent[i]] = count[i] / size
        # Write to file
        outputPath = self.imPath.split('/')[1].split('.')[0]
        file = open('cls_png_size_actual/' + outputPath + '.pkl', 'wb')
        pickle.dump(self.classCount, file)
        file.close()
        
        # Open
        #print(pickle.load(open('cls_png_size_actual/' + outputPath + '.pkl', 'rb')))


filesPng = glob.glob('cls_png/*.png')
for i in range(len(filesPng)):
    _ = Count(filesPng[i])