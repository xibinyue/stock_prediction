#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author:  xibin.yue   
date:  2016/12/21
descrption: 
"""
import pandas as pd
import os
import datetime


class DataGenerator(object):
    def __init__(self):
        self.data_path = 'data'
        self.time_format = '%Y-%m-%d'
        self.Y = pd.DataFrame()
        self.X = pd.DataFrame()
        self.DataSet = pd.DataFrame()
        self.main_price = None
        self.Factor = None

    def read_main_price(self):
        main_price = pd.read_json(os.path.join(self.data_path, 'price.json'))
        del main_price['exchangeCD'], main_price['secShortName'], main_price['secID']
        main_price['tradeDate'] = main_price['tradeDate'].map(
            lambda x: datetime.datetime.strptime(str(x), self.time_format))
        self.main_price = main_price

    def read_factor(self):
        equ_flow = pd.read_json(os.path.join(self.data_path, 'equFlow.json'))
        equ_flow.loc[:, equ_flow.columns != 'tradeDate'] /= 10 ** 8
        factor = pd.read_json(os.path.join(self.data_path, 'factor.json'))
        factor['MoneyFlow20'] /= 10**10
        self.Factor = pd.merge(equ_flow, factor, on='tradeDate')
        self.Factor['tradeDate'] = self.Factor['tradeDate'].map(
            lambda x: datetime.datetime.strptime(str(x), self.time_format))
        del self.Factor['ticker']

    def generate_data(self):
        dataset = pd.DataFrame()
        self.read_factor()
        self.read_main_price()
        dataset['tradeDate'] = self.main_price['tradeDate']
        dataset['chgPct'] = self.main_price['chgPct']
        dataset['tradeDate'] = dataset['tradeDate'].map(lambda x: x - datetime.timedelta(days=1))
        self.DataSet = pd.merge(self.Factor, dataset, on='tradeDate')
        del self.DataSet['tradeDate']
        self.DataSet['inflowS'] /= 10 ** 8
        self.DataSet['moneyInflow'] /= 10 ** 9
        self.DataSet['moneyOutflow'] /= 10 ** 9
        self.DataSet['netInflowS'] /= 10 ** 7
        self.DataSet['MoneyFlow20'] /= 10 ** 10


if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_data()
