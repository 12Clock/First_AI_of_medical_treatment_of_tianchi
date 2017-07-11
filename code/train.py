#coding:utf-8
from precondition import nodules_crop_3D
from model import _3D_CNN_2
from deal_data import _3D_data
from save_result import save_results
import numpy as np
from keras.optimizers import SGD
from keras import backend as K

import warnings
warnings.filterwarnings("ignore")

def get_data2(datas, size):
    new_datas = []
    size = np.array(size) / np.array([2,2,2])
    for data in datas:
        data = data[0]
        new_datas.append([data[16-size[0]:16+size[0],
                          16-size[1]:16+size[1],
                          16-size[2]:16+size[2]]])
    return np.array(new_datas)

def deal_datas(datas):
    datas = np.array(datas, dtype='float32').transpose((0, 2, 3, 4, 1))
    datas = (datas + 1000) / 1400
    return datas

if __name__ == '__main__':
    data = _3D_data(path='../nodule_cubes/')

    K.clear_session()
    weights_path = './weights/weights3.h5';
    model = _3D_CNN_2(dropout=True)
    model.setting(optimizer=SGD(lr=0.017, momentum=0.001, decay=1e-7, nesterov=False),
                  loss=["binary_crossentropy"], metrics=['accuracy'])

    (x_val32, y_val) = data.get_valset()
    x_val16 = (deal_datas(get_data2(x_val32, [16, 16, 16])))
    x_val32 = (deal_datas(x_val32))
    y_val = np.array(y_val, dtype='float32')

    model.load_data(x_val16=x_val16, x_val32=x_val32, y_val=y_val)

    for i in range(1000):
        print "[ %04d / %04d ] :" % (i + 1, 1000)

        (x_train32, y_train) = data.get_trainset(rate=1)
        x_train16 = (deal_datas(get_data2(x_train32, [16, 16, 16])))
        x_train32 = (deal_datas(x_train32))
        y_train = np.array(y_train, dtype='float32')

        model.load_data(x_train16=x_train16, x_train32=x_train32, y_train=y_train)

        model.train(batch_size=120, epochs=1, saveName='./weights/weights' + str(i + 1))
        print "---------------------------------------------------------------------------------------"