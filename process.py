import os
import numpy as np
import struct
from PIL import Image


data_dir = './data'   # 当前目录下新建一个data文件夹啊
train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')  # data文件夹下放这两个文件夹即可
test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield image, tagcode
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode, f.read(2)
char_set = set()
for _, tagcode, name in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
# print len(char_dict)
import pickle
f = open('char_dict', 'wb')
pickle.dump(char_dict, f)
f.close()
train_counter = 0
test_counter = 0
for image, tagcode, name in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    im = Image.fromarray(image)
    dir_name = './data/train/' + name  #'%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(train_counter) + '.png')
    train_counter += 1
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    im = Image.fromarray(image)
    dir_name = './data/test/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(test_counter) + '.png')
    test_counter += 1