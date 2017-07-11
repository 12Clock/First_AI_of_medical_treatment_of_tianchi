#coding:utf-8

from deal_data import _3D_data
from model import _3D_CNN_1
import numpy as np
from tqdm import tqdm
from save_result import save_results
import  pickle as pkl
import random
import scipy.ndimage

weights_path = './weights/weights30.h5'

def prec(index, size, labels, origins, new_spacings, stride=[32, 32, 32], batch_size = 250, position=None):

    if position is None:
        position = []
    n = 0
    x_vals = []
    centers = []
    val_labels = []
    tmp = {}
    position_num = 0
    negetive_num = 0

    for i in tqdm(index):
        t = valdata[i]
        # print '-----------------[ %3d / %3d ]-----------------------' % (i, len(valdata))
        for z in range(30, t.shape[0] - 62, stride[0]):
            # print '[ %3d / %3d ] :' % ((z-30)/4, (t.shape[0]-92)/4)
            for y in range(50, t.shape[1] - 82, stride[1]):
                for x in range(50, t.shape[2] - 82, stride[2]):
                    n = n + 1
                    x_val = np.array([t[z:z + size[0], y:y + size[1], x:x + size[2]]])
                    if x_val.shape != (1, 32, 32, 32):
                        x_val = scipy.ndimage.interpolation.zoom(x_val, np.array([1, 32.0, 32.0, 32.0]) / x_val.shape,
                                                         mode='nearest')
                    center = [z + size[0]/2, y + size[1]/2, x + size[2]/2]
                    x_vals.append(x_val)
                    centers.append(center)
                    val_labels.append(labels[i])
                    if n == batch_size:
                        y_vals = model.predict(
                            x_test=(np.array(x_vals, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400,
                            batch_size=batch_size)
                        for k in range(n):
                            if y_vals[k, 0] > 0.5:
                                center = centers[k] * new_spacings[val_labels[k]] + origins[val_labels[k]]
                                tmp['coordX'] = center[2]
                                tmp['coordY'] = center[1]
                                tmp['coordZ'] = center[0]
                                tmp['seriesuid'] = val_labels[k]
                                tmp['probability'] = y_vals[k, 0]
                                position.append(tmp)
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
                center = centers[k] * new_spacings[val_labels[k]] + origins[val_labels[k]]
                tmp['coordX'] = center[2]
                tmp['coordY'] = center[1]
                tmp['coordZ'] = center[0]
                tmp['seriesuid'] = val_labels[k]
                tmp['probability'] = y_vals[k, 0]
                position.append(tmp)
                position_num += 1
            else:
                negetive_num += 1

    print '[%d / %d]' % (position_num, negetive_num)
    return position

if __name__ == '__main__':
    data = _3D_data(valpath='../nodule_cubes/val_data/')
    labels, valdata, origins, new_spacings, old_spacings = data.get_val()

    model = _3D_CNN_1(weights_path=weights_path)

    index = range(len(valdata))
    # random.shuffle(index)
    # index = index[0:20]

    position = prec(index, [32, 32, 32], labels, origins, new_spacings, stride=[32, 32, 32])
    position = prec(index, [14, 14, 14], labels, origins, new_spacings, stride=[14, 14, 14], position=position)
    reaults = save_results(csvfile_path='./val_csv/annotations.csv')

    reaults.write(position)