# Preprocessing code from dCMF paper Mariappan 2019
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

    def loading_sample(self):
        print("Loading data from data_dir: ", self.data_dir)
        num_folds = 1
        U1 = pkl.load(open(self.data_dir+"X_13.pkl", 'rb'))
        U2 = pkl.load(open(self.data_dir+"X_14.pkl", 'rb'))
        V1 = pkl.load(open(self.data_dir+"X_26.pkl", 'rb'))
        W1 = pkl.load(open(self.data_dir+"X_53.pkl", 'rb'))
        r_temp_dict = {}
        for fold_num in np.arange(1, num_folds+1):
            r_train = pkl.load(
                open(self.data_dir+'/X_12_train_fold_'+str(fold_num)+'.pkl', 'rb'))
            r_train_idx = pkl.load(
                open(self.data_dir+'/X_12_train_idx_'+str(fold_num)+'.pkl', 'rb'))
            r_test = pkl.load(
                open(self.data_dir+'/X_12_test_fold_'+str(fold_num)+'.pkl', 'rb'))
            r_test_idx = pkl.load(
                open(self.data_dir+'/X_12_test_idx_'+str(fold_num)+'.pkl', 'rb'))
            r_doublets = pkl.load(
                open(self.data_dir+'/R_doublets_'+str(fold_num)+'.pkl', 'rb'))
            r_temp_dict[fold_num] = {"Rtrain": r_train, "Rtrain_idx": r_train_idx,
                                     "Rtest": r_test, "Rtest_idx": r_test_idx, "Rdoublets": r_doublets}
            self.data_dict = {"U1": U1, "U2": U2,
                              "V1": V1, "W1": W1, "R": r_temp_dict}

    def loading_matrices(self):
        # user feature
        df_rf = pd.read_csv(self.data_dir+'rf_all_more.csv')
        # movie feature
        df_cf = pd.read_csv(self.data_dir+'cf_all_more.csv')

        print("Completed loading the data")
        print("df_rf.shape:  ", df_rf.shape)
        print("df_cf.shape: ", df_cf.shape)
        print("#")

        org_u = df_rf.to_numpy()
        org_v = df_cf.to_numpy()
        print("orgU.shape: ", org_u.shape)
        print("orgV.shape: ", org_v.shape)
        print("#")

        u_scaler = preprocessing.MaxAbsScaler()
        self.U = u_scaler.fit_transform(org_u)
        v_scaler = preprocessing.MaxAbsScaler()
        self.V = v_scaler.fit_transform(org_v)

        print("U.shape: ", self.U.shape)
        print("V.shape: ", self.V.shape)
        print("#")
        print("U: ", np.min(self.U), ", ", np.max(
            self.U), ", ", np.median(self.U))
        print("V: ", np.min(self.V), ", ", np.max(
            self.V), ", ", np.median(self.V))
        print("#")
