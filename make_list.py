'''
JPEGImages: 17126 Images
SegmentationClass: 2914 Images
SegmentationClassAug: 12032 Images
'''
import os
from glob import glob
from shutil import copyfile

class Make_List:
  def __init__(self):
    self.root = os.getcwd()
    self.JPEG_images = self.root + '/VOC2012/JPEGImages/'
    self.segmentation_class = self.root + '/VOC2012/SegmentationClass/'
    self.segmentation_class_aug = self.root + '/VOC2012/SegmentationClassAug/'
    self.sbd_images = self.root + '/sbd/dataset/img/'
    self.sbd_aug = self.root + '/sbd/dataset/cls_png/'
    self.train = self.root + '/ImageSets/'

    # self.train_jsws()
    # self.val_jsws()
    # self.train_one_stage_full()
    # self.val_one_stage_full()
    # self.train_one_stage_original()
    # self.val_one_stage_original()
    # self.train_one_stage_sbd()
    self.val_one_stage_sbd()

  def make_folder(self, folder_name = 'train'):
    if not (os.path.exists(self.train + folder_name)):
      os.mkdir(self.train + folder_name)

  def segmentation_class_files(self, path):
    file_names = []
    files = glob(path + '*.png') #/Users/Sharhad/Desktop/VOCdevkit/VOC2012/SegmentationClassAug/2010_001767.png
    for file in files:
      file_names.append(file.split('/')[-1].split('.')[0])
    return file_names

  def check(self):
    images = []
    masks = []
    with open(self.train + 'one_stage/val_voc_original.txt', 'r') as lines:
      for line in lines:
        image, mask = line.strip('\n').split(' ')
        image = self.root + '/sbd/dataset/img/' + image.split('/')[-1].split('.')[0] + '.jpg'
        mask = self.root + '/sbd/dataset/cls_png/' + mask.split('/')[-1].split('.')[0] + '.png'
        if not os.path.isfile(image): images.append(image.split('/')[-1].split('.')[0])
        if not os.path.isfile(mask): masks.append(mask)
    print(images)

  def train_jsws(self, folder_name = 'jsws'):
    self.make_folder(folder_name = folder_name)
    files = self.segmentation_class_files(self.segmentation_class) + self.segmentation_class_files(self.segmentation_class_aug)
    text_file = open(self.train + folder_name + '/trainaug.txt', 'w')
    for file in files:
      text_file.write(file + '\n')
    text_file.close()

  def val_jsws(self, folder_name = 'jsws'):
    self.make_folder(folder_name = folder_name)
    files = self.segmentation_class_files(self.segmentation_class)
    text_file = open(self.train + folder_name + '/trainval.txt', 'w')
    for file in files:
      text_file.write(file + '\n')
    text_file.close()

  def train_one_stage_full(self, folder_name = 'one_stage'):
    self.make_folder(folder_name = folder_name)
    files_voc = self.segmentation_class_files(self.segmentation_class)
    files_aug = self.segmentation_class_files(self.segmentation_class_aug)
    for i, file in enumerate(files_voc):
      files_voc[i] = 'voc/VOCdevkit/VOC2012/JPEGImages/' + file + '.jpg' + ' voc/VOCdevkit/VOC2012/SegmentationClass/' + file + '.png' + '\n'
    for i, file in enumerate(files_aug):
      files_aug[i] = 'voc/VOCdevkit/VOC2012/JPEGImages/' + file + '.jpg' + ' voc/VOCdevkit/VOC2012/SegmentationClassAug/' + file + '.png' + '\n'
    files = files_voc + files_aug

    text_file = open(self.train + folder_name + '/train_augvoc.txt', 'w')
    for file in files:
      text_file.write(file)
    text_file.close()

  def val_one_stage_full(self, folder_name = 'one_stage'):
    self.make_folder(folder_name = folder_name)
    files = self.segmentation_class_files(self.segmentation_class)

    text_file = open(self.train + folder_name + '/val_voc.txt', 'w')
    for file in files:
      text_file.write('voc/VOCdevkit/VOC2012/JPEGImages/' + file + '.jpg' + ' voc/VOCdevkit/VOC2012/SegmentationClass/' + file + '.png' +'\n')
    text_file.close()

  def train_one_stage_original(self, txt_file = 'one_stage/train_augvoc_original.txt', folder_name = 'one_stage'):
    self.make_folder(folder_name = folder_name)
    text_file = open(self.train + folder_name + '/train_augvoc_converted.txt', 'w')
    with open(self.train + txt_file, 'r') as lines:
      for line in lines:
        image, mask = line.strip('\n').split(' ') #sbd/dataset/img/2008_007573.jpg sbd/dataset/cls_png/2008_007573.png
        assert os.path.isfile(image), '%s not found' % image
        assert os.path.isfile(mask), '%s not found' % mask

        image = 'VOC2012/JPEGImages/' + image.split('/')[-1].split('.')[0] + '.jpg'
        mask = 'VOC2012/SegmentationClassAug/' + mask.split('/')[-1].split('.')[0] + '.png'
        assert os.path.isfile(image), '%s not found' % image
        assert os.path.isfile(mask), '%s not found' % mask

        text_file.write('voc/VOCdevkit/' + image + ' voc/VOCdevkit/' + mask + '\n')
    text_file.close()

  def val_one_stage_original(self, folder_name = 'one_stage'):
    self.make_folder(folder_name = folder_name)
    copyfile(self.train + 'one_stage/val_voc_original.txt', self.train + 'one_stage/val_voc_converted.txt')

  def train_one_stage_sbd(self, folder_name = 'one_stage'):
    self.make_folder(folder_name = folder_name)
    files_sbd = self.segmentation_class_files(self.sbd_aug)
    text_file = open(self.train + folder_name + '/train_augvoc_sbd.txt', 'w')
    for file in files_sbd:
      image = self.sbd_images + file + '.jpg'
      mask = self.sbd_images + file + '.png'
      assert os.path.isfile(image), '%s not found' % image
      text_file.write('sbd/dataset/img/' + file + '.jpg' + ' sbd/dataset/cls_png/' + file + '.png' +'\n')
    text_file.close()

  def val_one_stage_sbd(self, folder_name = 'one_stage'):
    self.make_folder(folder_name = folder_name)
    text_file = open(self.train + folder_name + '/val_voc_sbd.txt', 'w')
    with open(self.train + 'one_stage/val_voc_original.txt', 'r') as lines:
      for line in lines:
        image, mask = line.strip('\n').split(' ')
        image = image.split('/')[-1].split('.')[0]
        mask = mask.split('/')[-1].split('.')[0]
        image_check = self.sbd_images + image + '.jpg'
        mask_check= self.sbd_aug + mask + '.png'
        if (os.path.isfile(image_check) and os.path.isfile(mask_check)):
          text_file.write('sbd/dataset/img/' + image + '.jpg' + ' sbd/dataset/cls_png/' + mask + '.png' +'\n')
    text_file.close()



Make_List()


#1. try with weight = 10
#
#2. update sizes

