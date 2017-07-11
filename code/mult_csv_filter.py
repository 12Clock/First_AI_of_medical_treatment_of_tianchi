#coding:utf-8

import multiprocessing
from csv_filter import _3D_csv_Filter

def init():
    global data
    from csv_filter import _3D_csv_Filter
    data = _3D_csv_Filter('../', type='new_test', size=5)

def save_mask(index):
    data.mult_save_mask(index)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=5, initializer=init)
    csv = _3D_csv_Filter('../')
    index = [0, 1, 2, 3, 4]
    pool.map(save_mask, index)