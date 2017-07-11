#coding:utf-8

from model import _3D_CNN_2
from deal_data import _3D_data
from save_result import save_feature
import pandas as pd
from tqdm import tqdm
import numpy as np

weights_path = './last_weights/weights86.h5'
data_type = 'train'
csv_path = '../nodule_cubes/'+data_type+'_data/csv/annotations.csv'
save_path = './feature/annotations.csv'
batch_size = 125
data_label = 0


def get_data_posotion(labels, key):
    for i in range(len(labels)):
        if key == labels[i]:
            return i
    return None

if __name__ == '__main__':

    model = _3D_CNN_2(weights_path=weights_path, dropout=True, feature=True)
    data_path = '../nodule_cubes/' + data_type + '_data/'

    if data_type == 'val':
        data = _3D_data(valpath=data_path)
        labels, valdata, origins, new_spacings, old_spacings = data.get_val()
    elif data_type == 'test' or data_type == 'new_test':
        data = _3D_data(testpath=data_path)
        labels, valdata, origins, new_spacings, old_spacings = data.get_test()
    elif data_type == 'train':
        data = _3D_data(trainpath=data_path)
        labels, valdata, origins, new_spacings, old_spacings = data.get_train()
    else:
        print 'ERROR!'

    csv_data = pd.read_csv(csv_path)
    names = csv_data['seriesuid']
    names = list(names)
    index = [get_data_posotion(labels, name) for name in names]

    n = 0
    x_val16s = []
    x_val32s = []
    y_vals = []
    names = []
    centers = []
    tmp = {}
    features = []

    for idx in tqdm(range(len(index))):

        v_center = [csv_data['coordZ'][idx],
                  csv_data['coordY'][idx],
                  csv_data['coordX'][idx]]

        i = index[idx]
        if i is None:
            continue
        name = labels[i]
        spacing = new_spacings[name]
        origin = origins[name]
        t = valdata[i]


        center = (np.array(v_center, dtype='float') - origin) / spacing

        x_val16 = t[int(center[0]) - 8:int(center[0]) + 8, int(center[1]) - 8:int(center[1]) + 8, int(center[2]) - 8:int(center[2]) + 8]
        if x_val16.shape != (16,16,16):
            continue
        x_val32 = t[int(center[0]) - 16:int(center[0]) + 16, int(center[1]) - 16:int(center[1]) + 16, int(center[2]) - 16:int(center[2]) + 16]
        if x_val32.shape != (32,32,32):
            continue

        names.append(name)
        centers.append(v_center)
        x_val16s.append([x_val16])
        x_val32s.append([x_val32])
        n += 1

        if n == batch_size:
            x_vals1 = (np.array(x_val16s, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
            x_vals2 = (np.array(x_val32s, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
            y_vals = model.get_feature(x_test16=x_vals1, x_test32=x_vals2, batch_size=batch_size)

            for k in range(n):
                tmp = {}
                tmp['seriesuid'] = ('%s_x%.1fy%.1fz%.1f' % (names[k], float(centers[k][2]), float(centers[k][1]), float(centers[k][0])))
                for f in range(32):
                    tmp['F_'+str(f+1)] = y_vals[k,0,0,0,f]
                tmp['label'] = data_label
                features.append(tmp)

            n = 0
            x_val16s = []
            x_val32s = []
            centers = []
            names = []

    if n > 0:
        x_vals1 = (np.array(x_val16s, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
        x_vals2 = (np.array(x_val32s, dtype='float32').transpose((0, 2, 3, 4, 1)) + 1000) / 1400
        y_vals = model.get_feature(x_test16=x_vals1, x_test32=x_vals2, batch_size=batch_size)

        for k in range(n):
            tmp = {}
            tmp['seriesuid'] = ('%s_x%.1fy%.1fz%.1f' % (names[k], float(centers[k][2]), float(centers[k][1]), float(centers[k][0])))
            for f in range(32):
                tmp['F_' + str(f + 1)] = y_vals[k, 0, 0, 0, f]
            tmp['label'] = data_label
            features.append(tmp)


    csv = save_feature(save_path)
    csv.write(features)





