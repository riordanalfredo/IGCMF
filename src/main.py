#!/usr/bin/env python
from preprocessing import Preprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
#
import torch
from torch.autograd import Variable
from torch.nn import Tanh, Sigmoid, ReLU, LeakyReLU
from torch import nn

#
import pickle as pkl
#
from sklearn import preprocessing
from functions import matrix_labels
#
import matplotlib
matplotlib.use('agg')

if __name__ == '__main__':
    DATA_DIR = '../../data/sample_data/'
    p = Preprocess(DATA_DIR)
    p.loading_sample()
    sample_matrix = p.data_dict['R'][1]['Rtrain']
    matrix_labels(sample_matrix)
    
