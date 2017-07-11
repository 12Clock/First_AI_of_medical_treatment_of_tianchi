#coding:utf-8

import multiprocessing
from precondition import nodules_crop_3D

def init():
    global data
    from precondition import nodules_crop_3D
    data = nodules_crop_3D('../')

def random_crop(mhd):
    data.random_crop(mhd)

def annotations_crop(type):
    data.annotations_crop(type)

if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=5, initializer=init)
    #print 'negative data:'
    #num = [20000,20000,20000,20000,20000]
    #pool.map(random_crop, num)
    print 'positive data:'
    type = ['xMirror','yMirror','RandomTrans','RandomTrans','RandomTrans']
    pool.map(annotations_crop, type)