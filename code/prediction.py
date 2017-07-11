#coding:utf-8

from deal_data import _3D_data
from model import _3D_CNN_1,_3D_CNN_2
import numpy as np
from tqdm import tqdm
from save_result import save_results
import pandas as pd
import scipy.ndimage

class _3D_prediction_1(object):

    def __init__(self, weights_path, dropout=False, data_type='val', data_path=None, csv=False):
        self.weights_path = weights_path
        self.dropout = dropout
        self.position = []
        self.data_type = data_type
        self.csv = csv
        if data_path is None:
            self.data_path = '../nodule_cubes/' + data_type + '_data/'
        else:
            self.data_path = data_path

        if data_type == 'val':
            self.data = _3D_data(valpath=data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_val()
        elif data_type == 'test' or data_type == 'new_test':
            self.data = _3D_data(testpath=data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_test()
        else:
            print 'ERROR!'

        if csv:
            self.csv_data = pd.read_csv(self.data_path + 'csv/annotations.csv')
            names = self.csv_data['seriesuid']
            names = list(set(names))
            self.index = [self.get_data_posotion(name) for name in names]
        else:
            self.index = range(len(self.valdata))

    def load_data(self, index=None, labels=None, origins=None, new_spacings=None, old_spacings=None):
        if index:
            self.index = index
        if labels:
            self.labels = labels
        if origins:
            self.origins = origins
        if new_spacings:
            self.new_spacings = new_spacings
        if old_spacings:
            self.old_spacings = old_spacings

    def prec(self, size, stride=[32, 32, 32], batch_size = 250, position=None):

        model = _3D_CNN_1(weights_path=self.weights_path, dropout=self.dropout)

        if position:
            self.position = position


        n = 0
        x_vals = []
        centers = []
        val_labels = []
        tmp = {}
        position_num = 0
        negetive_num = 0
        valdatas = []
        if size != [32, 32, 32]:
            print 'init:'
            for i in tqdm(self.index):
                valdatas.append(scipy.ndimage.interpolation.zoom(valdatas[i], np.array([32.0, 32.0, 32.0]) / np.array(size), mode='nearest'))

        if self.csv:
            for i in range(len(self.csv_data['seriesuid'])):

                n += 1
                name = self.csv_data['seriesuid'][i]
                val_labels.append(name)
                key = self.get_data_posotion(self.csv_data['seriesuid'][i])

                x = self.csv_data['coordX'][i]
                y = self.csv_data['coordY'][i]
                z = self.csv_data['coordZ'][i]
                center = np.array([z, y, x])
                centers.append(center)
                valdata = valdatas[key]
                center = (center * self.new_spacings[name] + self.origins[name]).astype(int)

                t = valdata[center[0] - 16:center[0] + 16,
                    center[1] - 16:center[1] + 16,
                    center[2] - 16:center[2] + 16]
                x_vals.append([t])

                if n == batch_size:
                    y_vals = model.predict(
                        x_test=(np.array(x_vals, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400,
                        batch_size=batch_size)
                    for k in range(n):
                        if y_vals[k, 0] > 0.5:
                            tmp['coordX'] = centers[k][2]
                            tmp['coordY'] = centers[k][1]
                            tmp['coordZ'] = centers[k][0]
                            tmp['seriesuid'] = val_labels[k]
                            tmp['probability'] = y_vals[k, 0]
                            self.position.append(tmp)
                            tmp = {}
                            position_num += 1
                        else:
                            negetive_num += 1
                    n = 0
                    x_vals = []
                    centers = []
                    val_labels = []

            if n > 0:
                y_vals = model.predict(
                    x_test=(np.array(x_vals, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400,
                    batch_size=batch_size)
                for k in range(n):
                    if y_vals[k, 0] > 0.5:
                        tmp['coordX'] = centers[k][2]
                        tmp['coordY'] = centers[k][1]
                        tmp['coordZ'] = centers[k][0]
                        tmp['seriesuid'] = val_labels[k]
                        tmp['probability'] = y_vals[k, 0]
                        self.position.append(tmp)
                        tmp = {}
                        position_num += 1
                    else:
                        negetive_num += 1

        else:
            for i in tqdm(self.index):
                t = valdatas[i]
                if size != [32, 32, 32]:
                    stride = np.array(stride) / np.array(size) * np.array([32, 32, 32])
                    stride = stride.astype(int)
                # print '-----------------[ %3d / %3d ]-----------------------' % (i, len(valdata))
                for z in range(30, t.shape[0] - 62, stride[0]):
                    # print '[ %3d / %3d ] :' % ((z-30)/4, (t.shape[0]-92)/4)
                    for y in range(50, t.shape[1] - 82, stride[1]):
                        for x in range(50, t.shape[2] - 82, stride[2]):
                            n = n + 1
                            x_val = np.array([t[z:z + 32, y:y + 32, x:x + 32]])
                            center = [z + 16, y + 16, x + 16]
                            x_vals.append(x_val)
                            centers.append(center)
                            val_labels.append(self.labels[i])
                            if n == batch_size:
                                y_vals = model.predict(
                                    x_test=(np.array(x_vals, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400,
                                    batch_size=batch_size)
                                for k in range(n):
                                    if y_vals[k, 0] > 0.5:
                                        center = centers[k] * self.new_spacings[val_labels[k]] + self.origins[val_labels[k]]
                                        tmp['coordX'] = center[2]
                                        tmp['coordY'] = center[1]
                                        tmp['coordZ'] = center[0]
                                        tmp['seriesuid'] = val_labels[k]
                                        tmp['probability'] = y_vals[k, 0]
                                        self.position.append(tmp)
                                        tmp = {}
                                        position_num += 1
                                    else:
                                        negetive_num += 1
                                n = 0
                                x_vals = []
                                centers = []
                                val_labels = []

            if n > 0:
                y_vals = model.predict(x_test=(np.array(x_vals, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400,
                                       batch_size=n)
                for k in range(n):
                    if y_vals[k, 0] > 0.5:
                        center = centers[k] * self.new_spacings[val_labels[k]] + self.origins[val_labels[k]]
                        tmp['coordX'] = center[2]
                        tmp['coordY'] = center[1]
                        tmp['coordZ'] = center[0]
                        tmp['seriesuid'] = val_labels[k]
                        tmp['probability'] = y_vals[k, 0]
                        self.position.append(tmp)
                        position_num += 1
                    else:
                        negetive_num += 1

        print '[%d / %d]' % (position_num, negetive_num)
        return self.position

    def get_data_posotion(self, key):
        for i in range(len(self.labels)):
            if key == self.labels[i]:
                return i


    def save_position(self, path=None):
        if path is None:
            path = './' + self.data_type + '_csv/annotations.csv'
        res = save_results(path)
        res.write(self.position)


class _3D_prediction_2(object):

    def __init__(self, weights_path, dropout=False, data_type='val', data_path=None, csv=False):
        self.weights_path = weights_path
        self.dropout = dropout
        self.position = []
        self.data_type = data_type
        self.csv = csv
        if data_path is None:
            self.data_path = '../nodule_cubes/' + data_type + '_data/'
        else:
            self.data_path = data_path

        if data_type == 'val':
            self.data = _3D_data(valpath=self.data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_val()
        elif data_type == 'test' or data_type == 'new_test':
            self.data = _3D_data(testpath=self.data_path)
            self.labels, self.valdata, self.origins, self.new_spacings, self.old_spacings = self.data.get_test()
        else:
            print 'ERROR!'

        if csv:
            self.csv_data = pd.read_csv(self.data_path + 'csv/annotations.csv')
            names = self.csv_data['seriesuid']
            names = list(names)
            self.index = [self.get_data_posotion(name) for name in names]
        else:
            self.index = range(len(self.valdata))

    def load_data(self, index=None, labels=None, origins=None, new_spacings=None, old_spacings=None):
        if index:
            self.index = index
        if labels:
            self.labels = labels
        if origins:
            self.origins = origins
        if new_spacings:
            self.new_spacings = new_spacings
        if old_spacings:
            self.old_spacings = old_spacings

    def prec(self, stride=[32, 32, 32], batch_size = 120, position=None):

        model = _3D_CNN_2(weights_path=self.weights_path, dropout=self.dropout)

        if position:
            self.position = position


        n = 0
        x_vals1 = []
        x_vals2 = []
        centers = []
        val_labels = []
        tmp = {}
        position_num = 0
        negetive_num = 0

        if self.csv:
            for i in tqdm(range(len(self.index))):

                v_center = np.array([self.csv_data['coordZ'][i],
                                     self.csv_data['coordY'][i],
                                     self.csv_data['coordX'][i]])
                i= self.index[i]
                t = self.valdata[i]
                name = self.labels[i]
                origin = self.origins[name]
                spacing = self.new_spacings[name]

                v_center = (v_center - origin) / spacing
                v_center = v_center - np.array([2, 3, 3])

                for z in range(int(v_center[0]), int(v_center[0]) + 4):
                    for y in range(int(v_center[1]), int(v_center[1]) + 6):
                        for x in range(int(v_center[2]), int(v_center[2]) + 6):
                            n = n + 1
                            x_val1 = np.array([t[z - 8:z + 8, y - 8:y + 8, x - 8:x + 8]])
                            x_val2 = np.array([t[z - 16:z + 16, y - 16:y + 16, x - 16:x + 16]])
                            center = [z, y, x]
                            x_vals1.append(x_val1)
                            x_vals2.append(x_val2)
                            centers.append(center)
                            val_labels.append(self.labels[i])
                            if n == batch_size:
                                x_vals1 = (np.array(x_vals1, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                                x_vals2 = (np.array(x_vals2, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                                y_vals = model.predict(x_test16=x_vals1, x_test32=x_vals2, batch_size=batch_size)
                                for k in range(n):
                                    if y_vals[k, 0] > 0.5:
                                        center = centers[k] * self.new_spacings[val_labels[k]] + self.origins[val_labels[k]]
                                        tmp['coordX'] = center[2]
                                        tmp['coordY'] = center[1]
                                        tmp['coordZ'] = center[0]
                                        tmp['seriesuid'] = val_labels[k]
                                        tmp['probability'] = y_vals[k, 0]
                                        self.position.append(tmp)
                                        tmp = {}
                                        position_num += 1
                                    else:
                                        negetive_num += 1
                                n = 0
                                x_vals1 = []
                                x_vals2 = []
                                centers = []
                                val_labels = []

            if n > 0:
                x_vals1 = (np.array(x_vals1, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                x_vals2 = (np.array(x_vals2, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                y_vals = model.predict(x_test16=x_vals1, x_test32=x_vals2, batch_size=batch_size)
                for k in range(n):
                    if y_vals[k, 0] > 0.5:
                        center = centers[k] * self.new_spacings[val_labels[k]] + self.origins[val_labels[k]]
                        tmp['coordX'] = center[2]
                        tmp['coordY'] = center[1]
                        tmp['coordZ'] = center[0]
                        tmp['seriesuid'] = val_labels[k]
                        tmp['probability'] = y_vals[k, 0]
                        self.position.append(tmp)
                        position_num += 1
                    else:
                        negetive_num += 1



        else:
            for i in tqdm(self.index):
                t = self.valdata[i]
                # print '-----------------[ %3d / %3d ]-----------------------' % (i, len(valdata))
                for z in range(30, t.shape[0] - 62, stride[0]):
                    # print '[ %3d / %3d ] :' % ((z-30)/4, (t.shape[0]-92)/4)
                    for y in range(50, t.shape[1] - 82, stride[1]):
                        for x in range(50, t.shape[2] - 82, stride[2]):
                            n = n + 1
                            x_val1 = np.array([t[z+8:z+24,y+8:y+24,x+8:x+24]])
                            x_val2 = np.array([t[z:z + 32, y:y + 32, x:x + 32]])
                            center = [z + 16, y + 16, x + 16]
                            x_vals1.append(x_val1)
                            x_vals2.append(x_val2)
                            centers.append(center)
                            val_labels.append(self.labels[i])
                            if n == batch_size:
                                x_vals1 = (np.array(x_vals1, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                                x_vals2 = (np.array(x_vals2, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                                y_vals = model.predict(x_test16=x_vals1, x_test32=x_vals2, batch_size=batch_size)
                                for k in range(n):
                                    if y_vals[k, 0] > 0.5:
                                        center = centers[k] * self.new_spacings[val_labels[k]] + self.origins[val_labels[k]]
                                        tmp['coordX'] = center[2]
                                        tmp['coordY'] = center[1]
                                        tmp['coordZ'] = center[0]
                                        tmp['seriesuid'] = val_labels[k]
                                        tmp['probability'] = y_vals[k, 0]
                                        self.position.append(tmp)
                                        tmp = {}
                                        position_num += 1
                                    else:
                                        negetive_num += 1
                                n = 0
                                x_vals1 = []
                                x_vals2 = []
                                centers = []
                                val_labels = []

            if n > 0:
                x_vals1 = (np.array(x_vals1, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                x_vals2 = (np.array(x_vals2, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
                y_vals = model.predict(x_test16=x_vals1, x_test32=x_vals2, batch_size=batch_size)
                for k in range(n):
                    if y_vals[k, 0] > 0.5:
                        center = centers[k] * self.new_spacings[val_labels[k]] + self.origins[val_labels[k]]
                        tmp['coordX'] = center[2]
                        tmp['coordY'] = center[1]
                        tmp['coordZ'] = center[0]
                        tmp['seriesuid'] = val_labels[k]
                        tmp['probability'] = y_vals[k, 0]
                        self.position.append(tmp)
                        position_num += 1
                    else:
                        negetive_num += 1

        print '[%d / %d]' % (position_num, negetive_num)
        return self.position

    def get_data_posotion(self, key):
        for i in range(len(self.labels)):
            if key == self.labels[i]:
                return i


    def save_position(self, path=None):
        if path is None:
            path = './' + self.data_type + '_csv/annotations.csv'
        res = save_results(path)
        res.write(self.position)
        print 'save successfully!'

if __name__ == '__main__':
    prec = _3D_prediction_2('./last_weights/weights86.h5', csv=True)
    position = prec.prec(stride=[4, 6, 6], batch_size=125)
    prec.save_position()
