# coding: utf-8

import csv

class save_results(object):

    def __init__(self, csvfile_path):

        self.csvfile_path = csvfile_path

    def write(self, data):
        with open(self.csvfile_path, 'wb') as file:
            file = csv.DictWriter(file, ['seriesuid','coordX','coordY','coordZ','probability'])
            file.writeheader()
            file.writerows(data)

class save_feature(object):

    def __init__(self, csvfile_path):

        self.csvfile_path = csvfile_path

    def write(self, data):
        with open(self.csvfile_path, 'wb') as file:

            label = ['seriesuid']
            for i in range(32):
                label.extend(['F_'+str(i+1)])
            label.extend(['label'])
            file = csv.DictWriter(file, label)
            file.writeheader()
            file.writerows(data)
            print 'save successfully !'