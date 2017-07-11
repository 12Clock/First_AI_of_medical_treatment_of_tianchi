# coding:utf-8

from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

import warnings

warnings.filterwarnings("ignore")


class _3D_CNN_1:
    def __init__(self, weights_path=None, dropout=False):

        inputs = Input(shape=(32, 32, 32, 1))
        conv1 = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(inputs)
        conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
        conv1 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(conv1)

        conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv1)
        conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
        if dropout:
            conv2 = Dropout(0.3)(conv2)

        conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv2)
        conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
        conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
        if dropout:
            conv3 = Dropout(0.4)(conv3)

        conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv3)
        conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
        conv4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv4)
        if dropout:
            conv4 = Dropout(0.5)(conv4)

        conv5 = Conv3D(64, (2, 2, 2), activation='relu')(conv4)

        conv6 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv5)
        # conv6 = Flatten()(conv6)

        conv7 = Conv3D(1, (1, 1, 1), activation=None)(conv6)
        conv7 = Flatten()(conv7)

        self.model = Model(input=inputs, output=conv7)

        if weights_path:
            self.model.load_weights(weights_path, by_name=False)

    def load_data(self, x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None):

        if x_train is not None:
            self.x_train = x_train
        if y_train is not None:
            self.y_train = y_train
        if x_val is not None:
            self.x_val = x_val
        if y_val is not None:
            self.y_val = y_val
        if x_test is not None:
            self.x_test = x_test
        if y_test is not None:
            self.y_test = y_test

    def setting(self, optimizer=SGD(lr=0.001, momentum=0.5, nesterov=True), loss=["binary_crossentropy"], metrics=["accuracy"]):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, batch_size=32, epochs=100, saveName=None):

        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(self.x_val, self.y_val))
        if saveName:
            self.model.save_weights(saveName + '.h5', overwrite=True)

    def predict(self, x_test=None, batch_size=1, verbose=0):

        if x_test is None:
            x_test = self.x_test

        self.y_test = self.model.predict(x_test, batch_size=batch_size, verbose=verbose)
        return self.y_test

    def save(self, saveName='test_weights'):
        self.model.save_weights(saveName + '.h5', overwrite=True)


class _3D_CNN_2:
    def __init__(self, weights_path=None, dropout=False, feature=False):

        inputs16 = Input(shape=(16, 16 ,16, 1), name='input16')
        conv16_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs16)
        #conv16_1 = Conv3D(32, (1, 1, 1), activation='relu', padding='same')(conv16_1)
        up16_1 = UpSampling3D((2, 2, 2))(conv16_1)

        inputs32 = Input(shape=(32, 32, 32, 1), name='input32')
        conv32_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs32)
        #conv32_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv32_1)

        mer2 = concatenate(inputs=[up16_1, conv32_1], axis=4)
        conv2 = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(mer2)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        conv2 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
        conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
        if dropout:
            conv3 = Dropout(0.3)(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        conv4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv4)
        if dropout:
            conv4 = Dropout(0.4)(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
        conv5 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv5)
        if dropout:
            conv5 = Dropout(0.5)(conv5)

        conv6 = Conv3D(64, (2, 2, 2), activation='relu')(conv5)

        conv7 = Conv3D(32, (1, 1, 1), activation='relu', name='feature')(conv6)
        # conv6 = Flatten()(conv6)

        conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
        conv8 = Flatten()(conv8)

        self.model = Model(inputs=[inputs16, inputs32], outputs=conv8)
        if weights_path:
            self.model.load_weights(weights_path, by_name=False)

        if feature:
            self.feature_model=Model(inputs=self.model.input, outputs=self.model.get_layer('feature').output)

    def load_data(self, x_train16=None, x_train32=None, y_train=None, x_val16=None,
                  x_val32=None, y_val=None, x_test16=None, x_test32=None, y_test=None):

        if x_train16 is not None:
            self.x_train16 = x_train16
        if x_train32 is not None:
            self.x_train32 = x_train32
        if y_train is not None:
            self.y_train = y_train
        if x_val16 is not None:
            self.x_val16 = x_val16
        if x_val32 is not None:
            self.x_val32 = x_val32
        if y_val is not None:
            self.y_val = y_val
        if x_test16 is not None:
            self.x_test16 = x_test16
        if x_test32 is not None:
            self.x_test32 = x_test32
        if y_test is not None:
            self.y_test = y_test

    def setting(self, optimizer=SGD(lr=0.001, momentum=0.5, nesterov=True), loss=["binary_crossentropy"], metrics=["accuracy"]):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, batch_size=32, epochs=100, saveName=None):

        self.model.fit({'input16':self.x_train16, 'input32':self.x_train32},
                       self.y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=({'input16':self.x_val16, 'input32':self.x_val32},
                                        self.y_val))
        if saveName:
            self.model.save_weights(saveName + '.h5', overwrite=True)

    def predict(self, x_test16=None, x_test32=None, batch_size=1, verbose=0):

        if x_test16 is None:
            x_test16 = self.x_test16
        if x_test32 is None:
            x_test32 = self.x_test32

        self.y_test = self.model.predict({'input16':x_test16, 'input32':x_test32}, batch_size=batch_size, verbose=verbose)
        return self.y_test

    def save(self, saveName='test_weights'):
        self.model.save_weights(saveName + '.h5', overwrite=True)

    def get_feature(self, x_test16=None, x_test32=None, batch_size=1, verbose=0):

        if x_test16 is None:
            x_test16 = self.x_test16
        if x_test32 is None:
            x_test32 = self.x_test32

        self.y_test = self.feature_model.predict({'input16': x_test16, 'input32': x_test32}, batch_size=batch_size, verbose=verbose)
        return self.y_test




