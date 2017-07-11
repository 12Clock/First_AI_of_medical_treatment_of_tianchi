#coding=utf-8

from __future__ import print_function, division
import SimpleITK as sitk
import math
import scipy.ndimage
import numpy as np
import csv
import cv2
import os
from glob import glob
import pandas as pd
from tqdm import tqdm # long waits are not fun
import random
import array
import pickle
    
import matplotlib.pyplot as plt

import warnings  # 不显示乱七八糟的warning
warnings.filterwarnings("ignore")
# import traceback

class nodules_crop_3D(object):
    def __init__(self, workspace, nodule_num=600, data_type="train"):
        """param: workspace: 本次比赛all_patients的父目录"""
        self.workspace = workspace
        self.all_patients_path = os.path.join(self.workspace,"nodule_cubes/"+data_type+"_data/npy/")
        self.nodules_npy_path = "../nodule_cubes/"+data_type+"/npy/"
        self.all_annotations_mhd_path = "../nodule_cubes/"+data_type+"/mhd/"
        self.ls_all_patients = glob(self.all_patients_path + "*.npy")
        self.ls_all_origin = "../nodule_cubes/"+data_type+"_data/pkl/origion.pkl"
        self.ls_all_new_spacing = "../nodule_cubes/" + data_type + "_data/pkl/new_spacing.pkl"
        self.df_annotations = pd.read_csv(self.workspace + "csv/"+data_type+"/annotations.csv")
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()
        self.nodule_num = nodule_num
    #---各种预定义
    def set_window_width(self, image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        image[image < MIN_BOUND] = MIN_BOUND
        image[image > MAX_BOUND] = MAX_BOUND
        return image
    #---设置窗宽
    def resample(self,image, old_spacing, new_spacing=[1, 1, 1]):
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing
    #---重采样
    def write_meta_header(self,filename, meta_dict):
        header = ''
        # do not use tags = meta_dict.keys() because the order of tags matters
        tags = ['ObjectType', 'NDims', 'BinaryData',
                'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
                'TransformMatrix', 'Offset', 'CenterOfRotation',
                'AnatomicalOrientation',
                'ElementSpacing',
                'DimSize',
                'ElementType',
                'ElementDataFile',
                'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
        for tag in tags:
            if tag in meta_dict.keys():
                header += '%s = %s\n' % (tag, meta_dict[tag])
        f = open(filename, 'w')
        f.write(header)
        f.close()
    def dump_raw_data(self,filename, data):
        """ Write the data into a raw format file. Big endian is always used. """
        #---将数据写入文件
        # Begin 3D fix
        data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
        # End 3D fix
        rawfile = open(filename, 'wb')
        a = array.array('f')
        for o in data:
            a.fromlist(list(o))
        # if is_little_endian():
        #    a.byteswap()
        a.tofile(rawfile)
        rawfile.close()
    def write_mhd_file(self,mhdfile, data, dsize):
        assert (mhdfile[-4:] == '.mhd')
        meta_dict = {}
        meta_dict['ObjectType'] = 'Image'
        meta_dict['BinaryData'] = 'True'
        meta_dict['BinaryDataByteOrderMSB'] = 'False'
        meta_dict['ElementType'] = 'MET_FLOAT'
        meta_dict['NDims'] = str(len(dsize))
        meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
        meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
        self.write_meta_header(mhdfile, meta_dict)
        pwd = os.path.split(mhdfile)[0]
        if pwd:
            data_file = pwd + '/' + meta_dict['ElementDataFile']
        else:
            data_file = meta_dict['ElementDataFile']
        self.dump_raw_data(data_file, data)
    def save_annotations_nodule(self, nodule_crop, z, y, x, data_type=1, deal_type='Normal', mhd=False, xR=None, yR=None, zR=None):
        if deal_type != 'RandomRota':
            dir_name = "%d_x%03dy%03dz%03d_%s_annotations" % (data_type, x, y, z, deal_type)
        else:
            dir_name = "%d_x%03dy%03dz%03d_%s_x%.2fy%.2fz%.2f_annotations" % (data_type, x, y, z, deal_type, xR, yR, zR)

        np.save(os.path.join(self.nodules_npy_path, dir_name + ".npy"), nodule_crop)
        # np.save(self.nodules_npy_path + str(1) + "_" + str(name_index) + '_annotations' + '.npy', nodule_crop)
        if mhd:
            self.write_mhd_file(self.all_annotations_mhd_path + dir_name + '.mhd', nodule_crop, nodule_crop.shape)
        self.nodule_num = self.nodule_num + 1
    #---保存结节文件，若需要使用Fiji软件查看分割效果可取消注释write_mhd_file


    def get_filename(self,file_list, case):
        for f in file_list:
            if case in f:
                return (f)
    #---匹配文件名
    def annotations_crop(self, type):

        Origin = np.load(self.ls_all_origin)  # ---获取“体素空间”中结节中心的坐标
        New_spacing = np.load(self.ls_all_new_spacing)
        for patient in enumerate(tqdm(self.ls_all_patients)):
            patient = patient[1]

            patient_nodules = self.df_annotations[self.df_annotations.file == patient]
            patient_name = patient_nodules.seriesuid.values[0]
            image = np.load(patient)
            origin = Origin[patient_name]
            new_spacing = New_spacing[patient_name]

            for index, nodule in patient_nodules.iterrows():
                nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])#---获取“世界空间”中结节中心的坐标
                v_center = np.rint((nodule_center - origin) / new_spacing)#映射到“体素空间”中的坐标
                n_center = np.array(v_center, dtype=int)
                #---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素

                v_diameter = 16
                x = None
                y = None
                z = None
                if type == 'RandomTrans':
                    n_center =n_center + np.array([random.randint(-2,2), random.randint(-4,4), random.randint(-4,4)])
                    img_crop = image[n_center[0] - v_diameter:n_center[0] + v_diameter,
                               n_center[1] - v_diameter:n_center[1] + v_diameter,
                               n_center[2] - v_diameter:n_center[2] + v_diameter]
                elif type == 'xMirror':
                    img_crop = image[n_center[0] - v_diameter:n_center[0] + v_diameter,
                               n_center[1] - v_diameter:n_center[1] + v_diameter,
                               n_center[2] - v_diameter:n_center[2] + v_diameter]
                    img_crop = self._3D_Mirror(img_crop,'x')
                elif type == 'yMirror':
                    img_crop = image[n_center[0] - v_diameter:n_center[0] + v_diameter,
                               n_center[1] - v_diameter:n_center[1] + v_diameter,
                               n_center[2] - v_diameter:n_center[2] + v_diameter]
                    img_crop = self._3D_Mirror(img_crop,'y')
                elif type == 'zMirror':
                    img_crop = image[n_center[0] - v_diameter:n_center[0] + v_diameter,
                               n_center[1] - v_diameter:n_center[1] + v_diameter,
                               n_center[2] - v_diameter:n_center[2] + v_diameter]
                    img_crop = self._3D_Mirror(img_crop,'z')
                elif type == 'RandomRota':
                    z = random.random() * 4 - 2
                    y = random.random() * 4 - 2
                    x = random.random() * 4 - 2
                    img_crop = image[n_center[0] - v_diameter:n_center[0] + v_diameter,
                               n_center[1] - v_diameter:n_center[1] + v_diameter,
                               n_center[2] - v_diameter:n_center[2] + v_diameter]
                    img_crop = self._3D_Rotation(img_crop, 'z', z)
                    img_crop = self._3D_Rotation(img_crop, 'y', y)
                    img_crop = self._3D_Rotation(img_crop, 'x', x)
                else:
                    img_crop = image[n_center[0] - v_diameter:n_center[0] + v_diameter,
                               n_center[1] - v_diameter:n_center[1] + v_diameter,
                               n_center[2] - v_diameter:n_center[2] + v_diameter]


                nodule_box = self.set_window_width(img_crop)
                n_center = n_center * new_spacing + origin

                self.save_annotations_nodule(nodule_box, n_center[0], n_center[1], n_center[2], data_type=1, deal_type=type, mhd=False, xR=x, yR=y, zR=z)


    
    def random_crop(self, num):

        for i in tqdm(range(num)):
            while True:
                index = random.randint(0,len(self.ls_all_patients)-1)
                patient = self.ls_all_patients[index]

                full_image_info = sitk.ReadImage(patient)
                full_scan = sitk.GetArrayFromImage(full_image_info)
                origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
                old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
                image, new_spacing = self.resample(full_scan, old_spacing)#---重采样

                random_z = random.randint(60,image.shape[0]-60)
                random_y = random.randint(60,image.shape[1]-60)
                random_x = random.randint(40,image.shape[2]-40)
                v_center = np.array([random_z, random_y, random_x], dtype=int)

                img_crop = image[v_center[0]-16:v_center[0]+16,
                           v_center[1] - 16:v_center[1] + 16,
                           v_center[2] - 16:v_center[2] + 16]#---截取立方体
                nodule_box = self.set_window_width(img_crop)#---设置窗宽，小于-1000的体素值设置为-1000

                if (nodule_box > -500).sum() > 1:
                    break

                v_center = v_center * new_spacing + origin

            self.save_annotations_nodule(nodule_box, v_center[0], v_center[1], v_center[2], type=0)



    def test_data(self, testpath):

        test_datas = []
        origins = {}
        new_spacings = {}
        old_spacings = {}

        testdirs = glob(testpath + '*.mhd')
        labels = [os.path.splitext(os.path.basename(testdir))[0] for testdir in testdirs]
        for i in tqdm(range(len(testdirs))):
            testdir = testdirs[i]
            full_image_info = sitk.ReadImage(testdir)
            full_scan = sitk.GetArrayFromImage(full_image_info)
            origin = np.array(full_image_info.GetOrigin())[::-1]  # ---获取“体素空间”中结节中心的坐标
            old_spacing = np.array(full_image_info.GetSpacing())[::-1]  # ---该CT在“世界空间”中各个方向上相邻单位的体素的间距
            image, new_spacing = self.resample(full_scan, old_spacing)
            np.save('../nodule_cubes/test_data/npy/'+labels[i]+'.npy', image)
            origins[labels[i]] = origin
            new_spacings[labels[i]] = new_spacing
            old_spacings[labels[i]] = old_spacing

        file = open('../nodule_cubes/test_data/pkl/' + 'origion.pkl','wb')
        pickle.dump(origins,file)
        file.close()

        file = open('../nodule_cubes/test_data/pkl/' + 'new_spacing.pkl', 'wb')
        pickle.dump(new_spacings, file)
        file.close()

        file = open('../nodule_cubes/test_data/pkl/' + 'old_spacing.pkl', 'wb')
        pickle.dump(old_spacings, file)
        file.close()

    def val_data(self, valpath):

        val_datas = []
        origins = {}
        new_spacings = {}
        old_spacings = {}

        valdirs = glob(valpath + '*.mhd')
        labels = [os.path.splitext(os.path.basename(valdir))[0] for valdir in valdirs]
        for i in tqdm(range(len(valdirs))):
            valdir = valdirs[i]
            full_image_info = sitk.ReadImage(valdir)
            full_scan = sitk.GetArrayFromImage(full_image_info)
            origin = np.array(full_image_info.GetOrigin())[::-1]  # ---获取“体素空间”中结节中心的坐标
            old_spacing = np.array(full_image_info.GetSpacing())[::-1]  # ---该CT在“世界空间”中各个方向上相邻单位的体素的间距
            image, new_spacing = self.resample(full_scan, old_spacing)
            np.save('../nodule_cubes/val_data/npy/'+labels[i]+'.npy', image)
            origins[labels[i]] = origin
            new_spacings[labels[i]] = new_spacing
            old_spacings[labels[i]] = old_spacing

        file = open('../nodule_cubes/val_data/pkl/' + 'origion.pkl','wb')
        pickle.dump(origins,file)
        file.close()

        file = open('../nodule_cubes/val_data/pkl/' + 'new_spacing.pkl', 'wb')
        pickle.dump(new_spacings, file)
        file.close()

        file = open('../nodule_cubes/val_data/pkl/' + 'old_spacing.pkl', 'wb')
        pickle.dump(old_spacings, file)
        file.close()

    def train_data(self, trainpath):

        train_datas = []
        origins = {}
        new_spacings = {}
        old_spacings = {}

        traindirs = glob(trainpath + '*.mhd')
        labels = [os.path.splitext(os.path.basename(traindir))[0] for traindir in traindirs]
        for i in tqdm(range(len(traindirs))):
            traindir = traindirs[i]
            full_image_info = sitk.ReadImage(traindir)
            full_scan = sitk.GetArrayFromImage(full_image_info)
            origin = np.array(full_image_info.GetOrigin())[::-1]  # ---获取“体素空间”中结节中心的坐标
            old_spacing = np.array(full_image_info.GetSpacing())[::-1]  # ---该CT在“世界空间”中各个方向上相邻单位的体素的间距
            image, new_spacing = self.resample(full_scan, old_spacing)
            np.save('../nodule_cubes/train_data/npy/'+labels[i]+'.npy', image)
            origins[labels[i]] = origin
            new_spacings[labels[i]] = new_spacing
            old_spacings[labels[i]] = old_spacing

        file = open('../nodule_cubes/train_data/pkl/' + 'origion.pkl','wb')
        pickle.dump(origins,file)
        file.close()

        file = open('../nodule_cubes/train_data/pkl/' + 'new_spacing.pkl', 'wb')
        pickle.dump(new_spacings, file)
        file.close()

        file = open('../nodule_cubes/train_data/pkl/' + 'old_spacing.pkl', 'wb')
        pickle.dump(old_spacings, file)
        file.close()

    def _3D_Mirror(self, image, type):
        list = []
        if type == 'z':
            for i in range(image.shape[0]):
                list.append(image[i,:,:])
            list.reverse()
            img = np.array(list)
        elif type == 'y':
            for i in range(image.shape[1]):
                list.append(image[:,i,:])
            list.reverse()
            img = np.array(list).transpose((1,0,2))
        else:
            for i in range(image.shape[2]):
                list.append(image[:,:,i])
            list.reverse()
            img = np.array(list).transpose((1,2,0))
        return img

    def _3D_Rotation(image,type,angle): 
        img_array = []
        if type == 'z':
            for i in range(image.shape[0]):
                img_array.append(scipy.ndimage.rotate(image[i,:,:],angle))
                img_array = np.array(img_array).transpose(0,1,2)
        elif type == 'y':
            for i in range(image.shape[1]):
                img_array.append(scipy.ndimage.rotate(image[:,i,:],angle))
                img_array = np.array(img_array).transpose(1,0,2)
        else:
            for i in range(image.shape[1]):
                img_array.append(scipy.ndimage.rotate(image[:,:,i],angle))
                img_array = np.array(img_array).transpose(2,0,1)
        return img_array