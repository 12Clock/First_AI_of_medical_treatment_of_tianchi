#coding:utf-8

import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

class Visual_data(object):

    def __init__(self, workspace='../', type='val'):

        self.workspace = workspace
        self.all_patients_path = os.path.join(self.workspace, 'nodule_cubes/' + type + "_data/npy/")
        self.tmp_workspace = os.path.join(self.workspace, "nodule_cubes/" + type + '_visual/')
        self.all_pkl_path = '../nodule_cubes/' + type + '_data/pkl/'
        self.ls_all_patients = glob(self.all_patients_path + "*.npy")
        self.df_annotations = pd.read_csv(self.workspace + 'csv/' + type + '/annotations.csv')
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()
        self.ds_annotations = pd.read_csv('./' + type + '_csv/annotations.csv')
        self.ds_annotations["file"] = self.ds_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.ds_annotations = self.ds_annotations.dropna()

    def normalize(self,image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):

        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    def set_window_width(self,image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):

        image[image > MAX_BOUND] = MAX_BOUND
        image[image < MIN_BOUND] = MIN_BOUND
        return image

    def get_filename(self,file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def get_mask(self, key, save=False):

        img_file = self.get_filename(self.ls_all_patients, key)
        mini_df = self.df_annotations[self.df_annotations["file"] == img_file]
        mini_ds = self.ds_annotations[self.ds_annotations['file'] == img_file]
        print 'mask : %d' % (len(mini_df))
        print 'prediction : %d' % (len(mini_ds))
        img_array = np.load(img_file)
        num_z, height, width = img_array.shape
        origin = np.load(self.all_pkl_path + 'origion.pkl')[key]
        spacing = np.load(self.all_pkl_path + 'new_spacing.pkl')[key]

        for node_idx, cur_row in tqdm(mini_df.iterrows()):
            node_X = cur_row["coordX"]
            node_Y = cur_row["coordY"]
            node_Z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            w_nodule_center = np.array([node_X, node_Y, node_Z])
            v_nodule_center = np.rint((w_nodule_center - origin) / spacing)

            i_Z = int(v_nodule_center[2])

            slice = img_array[i_Z].copy()
            # slice = scipy.ndimage.interpolation.zoom(slice, [0.5, 0.5], mode='nearest')
            slice = 255.0 * self.normalize(slice)
            slice = slice.astype(np.uint8)

            rgb_slice = cv2.cvtColor(slice,cv2.COLOR_GRAY2RGB)
            cv2.circle(rgb_slice,(int(v_nodule_center[0]),int(v_nodule_center[1])),int(diam)+5,[255,255,0],2)
            #rgb_slice = scipy.ndimage.interpolation.zoom(rgb_slice, [0.5, 0.5,1], mode='nearest')

            #nodule_mask = self.make_mask(w_nodule_center, i_z * spacing[2] + origin[2], width, height,spacing, origin, slice, diam=diam)
            #nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [0.5, 0.5], mode='nearest')
            #nodule_mask = 255.0 * nodule_mask
            #nodule_mask = nodule_mask.astype(np.uint8)

            #cv2.imwrite(os.path.join(self.tmp_workspace, "%s_mask_%04d_%04d.jpg" % (cur_row["seriesuid"], node_idx, i_z)), rgb_slice)

            for node_idx, cur_row in mini_ds.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]

                w_nodule_center = np.array([node_x, node_y, node_z])
                v_nodule_center = np.rint((w_nodule_center - origin) / spacing)

                i_z = int(v_nodule_center[2])

                if abs(i_z - i_Z) <= 2:

                    #rgb_slice = cv2.cvtColor(slice,cv2.COLOR_GRAY2RGB)
                    cv2.circle(rgb_slice,(int(v_nodule_center[0]),int(v_nodule_center[1])),16,[255, 0, 0],2)
                    #rgb_slice = scipy.ndimage.interpolation.zoom(rgb_slice, [0.5, 0.5, 1], mode='nearest')

                    # nodule_mask = self.make_mask(w_nodule_center, i_z * spacing[2] + origin[2], width, height, spacing,origin, slice)
                    # nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [0.5, 0.5], mode='nearest')
                    # nodule_mask = 255.0 * nodule_mask
                    # nodule_mask = nodule_mask.astype(np.uint8)

            plt.figure(key+"_"+str(i_Z)+"_Image")
            plt.imshow(rgb_slice)
            if save:
                plt.savefig(self.tmp_workspace + key + "_" + str(i_Z) + '.png')
            plt.show()
            plt.close()

if __name__ == '__main__':
    data = Visual_data()
    data.get_mask('LKDS-00409', save=True)
