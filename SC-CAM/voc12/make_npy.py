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
print(type(files))
print(files[24])

files = np.load('train_label.npy', allow_pickle = True)
print(type(files))
print(files[24])