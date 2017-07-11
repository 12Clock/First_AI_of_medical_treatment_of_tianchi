#coding = utf-8

import os
import numpy as np
from glob import glob
import random
from tqdm import tqdm
import pickle as pkl
from save_result import save_results

import warnings
warnings.filterwarnings("ignore")



class _3D_data(object):

    def __init__(self, testpath=None, path=None, valpath=None, trainpath=None):

        if path:
            self.positive_trainpath = glob(path + 'train/npy/' + '1_*.npy')
            self.negative_trainpath = glob(path + 'train/npy/' + '0_*.npy')
            self.valpath = glob(path + 'val/npy/' + '*.npy')
            self.new_negative_valpath = glob(path + 'new_val/npy/' + '0_*.npy')

            self.positive_traindata = []
            self.negative_traindata = []
            self.valdata = []
            self.vallabel = []
            self.new_negative_valdata = []
            self.new_negative_vallabel = []

            print 'train init:'

            #self.positive_trainpath = self.negative_trainpath[0:1]
            for i in tqdm(self.positive_trainpath):
                tmp = np.load(i)
                if type(tmp) != int and tmp.shape==(32,32,32):
                    self.positive_traindata.append(np.array([tmp.astype('float32')]))

            #self.negative_trainpath = self.negative_trainpath[0:1]
            for i in tqdm(self.negative_trainpath):
                self.negative_traindata.append(np.array([np.load(i).astype('float32')]))

            print 'val init:'

            #self.valpath = self.valpath[0:1]
            for i in tqdm(self.valpath):
                tmp = np.load(i)
                if type(tmp) != int and tmp.shape == (32, 32, 32):
                    self.valdata.append(np.array([tmp.astype('float32')]))
                    self.vallabel.append(1)

            print 'new val init:'

            #self.new_negative_valpath = self.new_negative_valpath[0:1]

            for i in tqdm(self.new_negative_valpath):
                self.new_negative_valdata.append(np.array([np.load(i).astype('float32')]))

        if testpath:

            self.testpath = glob(testpath+'npy/'+'*.npy')
            self.testdata = [np.load(t) for t in tqdm(self.testpath)]
            self.labels = [os.path.splitext((os.path.basename(t)))[0] for t in self.testpath]
            tmpdir = open(testpath+'pkl/origion.pkl','rb')
            self.origions = pkl.load(tmpdir)
            tmpdir.close()
            tmpdir = open(testpath + 'pkl/new_spacing.pkl', 'rb')
            self.new_spacings = pkl.load(tmpdir)
            tmpdir.close()
            tmpdir = open(testpath + 'pkl/old_spacing.pkl', 'rb')
            self.old_spacings = pkl.load(tmpdir)
            tmpdir.close()

        if valpath:

            self.valpath = glob(valpath+'npy/'+'*.npy')
            self.valdata = [np.load(t) for t in tqdm(self.valpath)]
            self.labels = [os.path.splitext((os.path.basename(t)))[0] for t in self.valpath]
            tmpdir = open(valpath+'pkl/origion.pkl','rb')
            self.origions = pkl.load(tmpdir)
            tmpdir.close()
            tmpdir = open(valpath + 'pkl/new_spacing.pkl', 'rb')
            self.new_spacings = pkl.load(tmpdir)
            tmpdir.close()
            tmpdir = open(valpath + 'pkl/old_spacing.pkl', 'rb')
            self.old_spacings = pkl.load(tmpdir)
            tmpdir.close()

        if trainpath:

            self.trainpath = glob(trainpath+'npy/'+'*.npy')[0:300]
            self.traindata = [np.load(t) for t in tqdm(self.trainpath)]
            self.labels = [os.path.splitext((os.path.basename(t)))[0] for t in self.trainpath]
            tmpdir = open(trainpath + 'pkl/origion.pkl', 'rb')
            self.origions = pkl.load(tmpdir)
            tmpdir.close()
            tmpdir = open(trainpath + 'pkl/new_spacing.pkl', 'rb')
            self.new_spacings = pkl.load(tmpdir)
            tmpdir.close()
            tmpdir = open(trainpath + 'pkl/old_spacing.pkl', 'rb')
            self.old_spacings = pkl.load(tmpdir)
            tmpdir.close()




    def get_trainset(self, rate=None, new_val=False):

        positive = len(self.positive_traindata)
        negative = len(self.negative_traindata)
        if rate:
            if positive <= negative:
                negative = positive / rate
            else:
                positive = negative * rate

        new_negative = min(negative * 3 / 5, len(self.new_negative_valdata))
        negative = negative - new_negative

        positivedata = []
        negativedata = []
        new_negativedata = []

        temp = range(len(self.positive_traindata))
        random.shuffle(temp)
        temp = temp[0:positive]
        for i in temp:
            positivedata.append(self.positive_traindata[i])

        temp = range(len(self.negative_traindata))
        random.shuffle(temp)
        temp = temp[0:negative]
        for i in temp:
            negativedata.append(self.negative_traindata[i])

        if new_val:
            temp = range(len(self.new_negative_valdata))
            random.shuffle(temp)
            temp = temp[0:new_negative]
            for i in temp:
                new_negativedata.append(self.new_negative_valdata[i])

            temp = range(positive + negative + new_negative)
            random.shuffle(temp)
            data = []
            label = []
            for i in temp:
                if i < positive:
                    data.append(positivedata[i])
                    label.append(1)
                elif i < positive + negative:
                    data.append(negativedata[i - positive])
                    label.append(0)
                else:
                    data.append(new_negativedata[i - positive - negative])
                    label.append(0)

        else:
            temp = range(positive + negative)
            random.shuffle(temp)
            data = []
            label = []
            for i in temp:
                if i < positive:
                    data.append(positivedata[i])
                    label.append(1)
                else:
                    data.append(negativedata[i - positive])
                    label.append(0)


        return (data, label)



    def get_valset(self):

        return (self.valdata, self.vallabel)

    def get_test(self):

        return self.labels, self.testdata, self.origions, self.new_spacings, self.old_spacings
    
    def get_val(self):

        return self.labels, self.valdata, self.origions, self.new_spacings, self.old_spacings

    def get_train(self):

        return self.labels, self.traindata, self.origions, self.new_spacings, self.old_spacings

    def get_Random_train(self, num, path='../nodule_cubes/train_data/csv/'):

        data = []

        for i in tqdm(range(num)):

            while True:
                t = random.randint(0, 299)
                CT = self.traindata[t]
                z = random.randint(30, CT.shape[0]-30)
                y = random.randint(60, CT.shape[1]-60)
                x = random.randint(60, CT.shape[2]-60)
                tmp = CT[z - 16:z + 16, y - 16:y + 16, x - 16:x + 16]

                if (tmp > -600).sum() > 5:
                    break

            v_center = np.array([z, y, x], dtype='float')
            v_center = v_center * self.new_spacings[self.labels[t]] + self.origions[self.labels[t]]

            data.append({'seriesuid': self.labels[t],
                     'coordX': v_center[2],
                     'coordY': v_center[1],
                     'coordZ': v_center[0],
                     'probability': 0.5})

        csv = save_results(path + 'annotations.csv')
        csv.write(data)






if __name__ == '__main__':

    data = _3D_data(trainpath='../nodule_cubes/train_data/')
    data.get_Random_train(10000)



