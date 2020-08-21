"""
    Preprocessing data. For this experiment, we will take data from ML100k, split into training, testing, and validation. For side features, we convert them into matrix that will be used for training process.
"""

import pandas as pd
import numpy as np
#
import pickle as pkl
#
from sklearn import preprocessing
#
import matplotlib
matplotlib.use('agg')


class Preprocess:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_dict = {}
