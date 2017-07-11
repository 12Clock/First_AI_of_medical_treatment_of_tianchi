#coding:utf-8

from prediction import _3D_prediction_2

weights_path = './last_weights/weights86.h5'

if __name__ == '__main__':

    pred = _3D_prediction_2(weights_path, dropout=True, data_type='new_test', csv=True)
    pred.prec(stride=[4, 6, 6], batch_size=125)
    pred.save_position()