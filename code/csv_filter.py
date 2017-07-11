#coding:utf-8

import numpy as np
from skimage import measure,morphology
import skimage
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from glob import glob
from save_result import save_results
import os

class _3D_csv_Filter(object):
    def __init__(self, workspace, type='val', size=2):
        self.workspace = workspace + 'nodule_cubes/'
        self.pkl_path = os.path.join(self.workspace, type + '_data/pkl/')
        self.npy_path = os.path.join(self.workspace, type + '_data/npy/')
        self.npy_path = glob(self.npy_path + '*.npy')
        self.csv_path = './' + type + '_csv/'
        self.mask_save = os.path.join(self.workspace + type + '_data/mask/')
        self.mask_path = glob(self.mask_save + '*.npy')
        self.csv_save_path = './new_' + type + '_csv/'
        self.csv_data1 = pd.read_csv(self.csv_path + 'annotations.csv')
        self.csv_data2 = pd.read_csv(self.csv_path + 'annotations.csv')
        self.csv_data1["file"] = self.csv_data1["seriesuid"].map(lambda file_name: self.get_filename(self.npy_path, file_name))
        self.csv_data2["file"] = self.csv_data2["seriesuid"].map(lambda file_name: self.get_filename(self.mask_path, file_name))
        self.csv_data1 = self.csv_data1.dropna()
        self.csv_data2 = self.csv_data2.dropna()
        self.mult_npy_path = []
        num = len(self.npy_path)/size
        for i in range(size-1):
            tmp = self.npy_path[i*num:(i+1)*num]
            self.mult_npy_path.append(tmp)
        self.mult_npy_path.append(self.npy_path[(size-1)*num:len(self.npy_path)])

    def get_filename(self,file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def get_mask(self, image, threshold=-450):
        mask = image > threshold
        return mask

    def filter_mask(self, mask):
        new_mask = np.zeros(mask.shape)
        measured = measure.label(mask)
        regions = measure.regionprops(measured)
        regions = sorted(regions, key=lambda x: x.area , reverse=True)

        region = regions[0]
        coords = region.coords
        for coord in coords:
            new_mask[coord[0], coord[1], coord[2]] = 1

        return new_mask

    def m_filter_mask(self, mask):

        new_mask = np.zeros_like(mask)
        morphology.binary_erosion(mask, selem=morphology.ball(3), out=new_mask)

        return new_mask

    def noduleFilter(self):
        New_spacing = np.load(self.pkl_path + 'new_spacing.pkl')
        Origion = np.load(self.pkl_path + 'origion.pkl')
        csvdir = save_results(self.csv_save_path + 'annotations.csv')
        result = []

        for patient in enumerate(tqdm(self.mask_path)):
            patient = patient[1]

            nodules_masks = self.csv_data2[self.csv_data2.file == patient]
            patient_name = nodules_masks.seriesuid.values[0]
            self.mask = np.load(patient)
            origion = Origion[patient_name]
            new_spacing = New_spacing[patient_name]

            for index, nodule in nodules_masks.iterrows():
                nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
                v_center = np.rint((nodule_center - origion) / new_spacing)
                v_center = np.array(v_center, dtype=int)

                if self.mask[v_center[0], v_center[1], v_center[2]] == 0:
                    tmp = {}
                    tmp["seriesuid"] = patient_name
                    tmp["coordX"] = nodule["coordX"]
                    tmp["coordY"] = nodule["coordY"]
                    tmp["coordZ"] = nodule["coordZ"]
                    tmp["probability"] = nodule["probability"]
                    result.append(tmp)

        csvdir.write(result)
        print ('Filter csv: %5d' % (len(result)))

    def save_mask(self):
        for patient in enumerate(tqdm(self.npy_path)):
            patient = patient[1]

            patient_nodules = self.csv_data1[self.csv_data1.file == patient]
            patient_name = patient_nodules.seriesuid.values[0]
            image = np.load(patient)
            self.mask = (self.get_mask(image))
            self.mask = self.filter_mask(self.mask)
            mask = self.mask
            for i in range(1):
                mask = self.m_filter_mask(mask)

            np.save(self.mask_save + patient_name + '.npy', mask)

    def mult_save_mask(self, index):

        for patient in enumerate(tqdm(self.mult_npy_path[index])):
            patient = patient[1]

            patient_nodules = self.csv_data1[self.csv_data1.file == patient]
            patient_name = patient_nodules.seriesuid.values[0]
            image = np.load(patient)
            self.mask = (self.get_mask(image))
            self.mask = self.filter_mask(self.mask)
            mask = self.mask
            for i in range(1):
               mask = self.m_filter_mask(mask)

            np.save(self.mask_save + patient_name + '.npy', mask)

if __name__ == '__main__':
    csv = _3D_csv_Filter('../', type='new_test', size=2)
    csv.noduleFilter()