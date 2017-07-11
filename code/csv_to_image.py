#coding:utf-8

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from glob import glob
from deal_data import _3D_data

class csv_to_image(object):
    def __init__(self, workspace='../', data_type='val'):

        self.workspace = workspace
        self.csvpath = './' + data_type +'_csv/annotations.csv'
        self.data_path = '../nodule_cubes/' + data_type + '_data/'
        self.savepath = self.data_path + 'image/'

        if data_type == 'val':
            self.data = _3D_data(valpath=self.data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_val()
        elif data_type == 'test' or data_type == 'new_test':
            self.data = _3D_data(testpath=self.data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_test()
        elif data_type == 'train':
            self.data = _3D_data(trainpath=self.data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_train()
        else:
            print 'ERROR!'

        self.csv_data = pd.read_csv(self.csvpath)
        names = self.csv_data['seriesuid']
        names = list(names)
        self.index = [self.get_data_posotion(name) for name in names]

    def get_data_posotion(self, key):
        for i in range(len(self.labels)):
            if key == self.labels[i]:
                return i

    def get_image(self, size=[32, 32]):

        index = range(len(self.csv_data))

        for i in tqdm(index):

            v_center = np.array([self.csv_data['coordZ'][i],
                                 self.csv_data['coordY'][i],
                                 self.csv_data['coordX'][i]])
            i = self.index[i]
            if i is None:
                continue
            t = self.valdata[i]
            name = self.labels[i]
            origin = self.origins[name]
            spacing = self.new_spacings[name]

            v_center = (v_center - origin) / spacing
            v_center = v_center - np.array([0, size[1]/2, size[0]/2])

            image = t[int(v_center[0]), int(v_center[1]):int(v_center[1]) + size[1], int(v_center[2]):int(v_center[2]) + size[0]]
            image = np.array(image, dtype='float').transpose((1, 0))
            image = image.reshape((size[0], size[1]))
            image = (image + 1000) / 1400
            image = image * 255
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            cv2.imwrite(self.savepath + '%s_x%.1fy%.1fz%.1f.png' % (name, v_center[2], v_center[1], v_center[0]), image)

if __name__ == '__main__':
    img = csv_to_image(data_type='new_test')
    img.get_image()


