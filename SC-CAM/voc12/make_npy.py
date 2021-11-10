import numpy as np
# files = np.load('cls_labels.npy', allow_pickle = True).item()
# temp = []

# for key, value in files.items():
#   temp.append(value)

# arr = np.asarray(temp)

# print(arr.shape)
# print(type(arr))

# np.save('train_label_all.npy', arr)

files = np.load('train_label_all.npy', allow_pickle = True)
print(files.shape)
print(type(files))
print(files[24])

files = np.load('train_label.npy', allow_pickle = True)
print(type(files))
print(files[24])

# temp = []
# files = np.load('cls_labels.npy', allow_pickle = True).item()
# with open('train_aug.txt', 'r') as lines:
#   for line in lines:
#     image, _ = line.strip('\n').split(' ')
#     img = image.split('/')[-1].split('.')[0]
#     temp.append(files['' + img])

# arr = np.asarray(temp)
# np.save('train_label_all.npy', arr)