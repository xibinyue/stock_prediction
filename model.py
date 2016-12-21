#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author:  xibin.yue   
date:  2016/12/21
descrption: 
"""
from dataset_io import DataGenerator
from sklearn.svm import SVR
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd
import numpy as np


class Model(object):
    def __init__(self):
        self.split_threshold = 0.8
        self.DataGenerator = DataGenerator()
        self.train_X = None
        self.train_Y = None
        self.valid_X = None
        self.valid_Y = None
        self.DataSet = None
        self.DataSetScaled = None
        self.par = ['rbf', 3.5, 0.0052, 0.01, 0.010]

    def generate_train_validation(self):
        self.DataGenerator.generate_data()
        self.DataSet = self.DataGenerator.DataSet
        self.normalization()
        self.train_Y = self.DataSetScaled['chgPct'].loc[0:self.split_threshold * len(self.DataSet)]
        self.valid_Y = self.DataSetScaled['chgPct'].loc[self.split_threshold * len(self.DataSet):]
        self.train_X = self.DataSetScaled.drop('chgPct', axis=1).loc[0:self.split_threshold * len(self.DataSet)]
        self.valid_X = self.DataSetScaled.drop('chgPct', axis=1).loc[self.split_threshold * len(self.DataSet):]
        assert len(self.train_X) == len(self.train_Y), 'Length-X And Length-Y Not Equal'

    def normalization(self):
        self.DataSetScaled = pd.DataFrame(preprocessing.scale(self.DataSet), index=self.DataSet.index,
                                          columns=self.DataSet.columns)

    def select_features(self, alpha=-1, threshold_q='mean'):
        fore_model = linear_model.RidgeCV(alphas=[alpha])
        fore_model.fit(self.train_X, np.ravel(np.array(self.train_Y), 1))
        sfm = feature_selection.SelectFromModel(fore_model, threshold=threshold_q, prefit=True)
        n_features = sfm.transform(self.train_X).shape[1]
        train_X_selected = pd.DataFrame(sfm.transform(self.train_X), index=self.train_X.index,
                                        columns=self.train_X.columns[sfm.get_support()])
        return train_X_selected, sfm

    def build_model(self):
        self.generate_train_validation()
        data_X, sfm = self.select_features()
        data_Y = self.train_Y
        model_ = SVR(self.par[0], C=self.par[1], gamma=self.par[2])
        try:
            model_.fit(np.array(data_X), np.ravel(np.array(data_Y), 1))
        except ValueError:
            print 'Value Error..'
        predict_y = model_.predict(data_X)
        mse = (
            (np.array(predict_y).reshape(len(predict_y), ) - np.array(data_Y).reshape(len(predict_y), )) ** 2)
        print mse
        return model_, sfm

    def predict(self):
        model_saved, sfm = self.build_model()
        vaild_X = sfm.transform(self.valid_X)
        predict_Y = model_saved.predict(vaild_X)
        print predict_Y, self.valid_Y


if __name__ == '__main__':
    model = Model()
    model.generate_train_validation()
    model.normalization()
    model.build_model()
    # model.predict()
