import numpy as np
import pickle as pkl
import time
import itertools
import pprint
import scipy
import torch
import os

from torch.autograd import Variable

from src.aec import autoencoder
from src.base import base

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


class dcmf_base(base):

    def __init__(self, G, X_data, X_meta, num_chunks,
                 k, kf, e_actf, d_actf,
                 learning_rate, weight_decay, convg_thres, max_epochs,
                 is_gpu=False, gpu_ids="1",
                 is_pretrain=False, pretrain_thres=None, max_pretrain_epochs=None,
                 is_linear_last_enc_layer=False, is_linear_last_dec_layer=False,
                 X_val={}, val_metric="rmse", at_k=10, is_val_transpose=False, num_folds=1):

        print("dcmf_base.__init__ - start")
        # outputs
        self.U_dict_ = {}
        self.X_prime_dict_ = {}
        # inputs
        self.G = G
        self.X_data = X_data
        self.X_meta = X_meta
        self.X_val = X_val
        # hyperparams
        # learning algo
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.convg_thres = convg_thres
        self.max_epochs = max_epochs
        self.is_pretrain = is_pretrain
        self.pretrain_thres = pretrain_thres
        self.max_pretrain_epochs = max_pretrain_epochs
        # data
        self.num_chunks = num_chunks
        # network
        self.k = k
        self.kf = kf
        self.e_actf = e_actf
        self.d_actf = d_actf
        self.is_linear_last_enc_layer = is_linear_last_enc_layer
        self.is_linear_last_dec_layer = is_linear_last_dec_layer
        self.E = len(G.keys())
        self.M = len(X_data.keys())
        # bookkeeping
        self.dict_epoch_loss = {}
        self.dict_epoch_aec_rec_loss = {}
        self.dict_epoch_mat_rec_loss = {}
        # gpu
        self.is_gpu = is_gpu
        self.gpu_ids = gpu_ids
        # val
        self.val_metric = val_metric
        self.at_k = at_k
        self.is_val_transpose = is_val_transpose
        # check type and format
        self.is_bo = False  # To perform validation accordingly
        self.is_dcmf_base = False
        self.num_folds = num_folds
        # set the gpu_id to use
        if self.is_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        # concatenated-matrix
        self.C_dict = {}
        self.C_dict_chunks = {}
        # pytorch variable version of the input matrices
        self.X_data_var = {}
        # Network
        self.N_aec_dict = {}
        # loss
        self.loss_list = []  # list of losses, loss/elements of the list corresponds to tasks
        self.dict_epoch_loss = {}
        self.dict_epoch_aec_rec_loss = {}
        self.dict_epoch_mat_rec_loss = {}
        # pretrain loss
        self.pretrain_dict_epoch_loss = {}
        self.pretrain_dict_epoch_aec_rec_loss = {}
        # validation set performance
        self.X_val_perf = {}
        self.pp = pprint.PrettyPrinter()
        print("dcmf_base.__init__ - end")

        # flag that says the call is from dcmf or dcmf_base
        self.validate_input()
        self.print_params()

    def __input_transformation(self):
        # for each entity, construct a concatenated matrix (as input to the corresponding autoencoder)
        # Building C_dict
        print("__input_transformation - start")
        print("#")
        print("concatenated-matrix construction...")
        for e_id in self.G.keys():
            print("e_id: ", e_id)
            X_id_list = self.G[e_id]
            print("X_id_list: ", X_id_list)
            X_data_list = []
            for X_id in X_id_list:
                print("X_id: ", X_id)
                print("X[X_id].shape: ", self.X_data[X_id].shape)
                if self.X_meta[X_id][0] == e_id:
                    X_data_list.append(self.X_data[X_id])
                else:
                    X_data_list.append(self.X_data[X_id].T)
            C_temp = Variable(torch.from_numpy(np.concatenate(
                X_data_list, axis=1)).float(), requires_grad=False)
            if self.is_gpu:
                self.C_dict[e_id] = C_temp.cuda()
            else:
                self.C_dict[e_id] = C_temp
            print("C_dict[e].shape: ", self.C_dict[e_id].shape)
            print("---")
        print("#")
        print("concatenated-matrix chunking...")
        # chunking
        # assert that num_chunks is not > number of datapoints
        # using the latest e_id to init
        min_num_datapoints = self.C_dict[e_id].shape[0]
        min_e_id_num_chunks = e_id
        # warn if the encoding length k is not > than minimum of the feature lengths
        # using the latest e_id to init
        min_features = self.C_dict[e_id].shape[0]
        min_e_id_k = e_id
        for e_id in self.G.keys():
            # num_chunks
            temp_num = self.C_dict[e_id].shape[0]
            if min_num_datapoints > temp_num:
                min_num_datapoints = temp_num
                min_e_id_num_chunks = e_id
            # k
            temp_feat = self.C_dict[e_id].shape[1]
            if min_features > temp_feat:
                min_features = temp_feat
                min_e_id_k = e_id
        assert (self.num_chunks <= min_num_datapoints), \
            "The num_chunks must be <= minimum entity size in the setting. Entity with ID: "+str(min_e_id_num_chunks) +\
            " is of minimum size "+str(min_num_datapoints)+". The num_chunks "+str(
            self.num_chunks)+" is larger than minimim entity size."
        if (self.k >= min_features):
            print("WARNING: Entity with ID: "+str(min_e_id_k) +
                  " has minimum feature size "+str(min_features)+" in the setting. The encoding length k "+str(self.k)+" is larger than minimim entity feature size.")
        print("#")
        print("e_id: ", min_e_id_num_chunks, ", min_num_datapoints: ",
              min_num_datapoints, ", num_chunks: ", self.num_chunks)
        print("e_id: ", min_e_id_k, ", min_features: ",
              min_features, ", k: ", self.k)
        print("#")
        # Building C_dict_chunks
        for e_id in self.C_dict.keys():
            print("e_id: ", e_id,
                  " C_dict[e_id].shape: ", self.C_dict[e_id].shape)
            C_temp = self.C_dict[e_id]
            C_temp_chunks_list = torch.chunk(C_temp, self.num_chunks, dim=0)
            print("C_temp_chunks_list[0].shape: ", C_temp_chunks_list[0].shape)
            self.C_dict_chunks[e_id] = C_temp_chunks_list
            print("---")
        print("#")
        print("creating pytorch variables of input matrices...")
        # Convert input matrices to pytorch variables (to calculate reconstruction loss)
        # Building X_data_var
        for X_id in self.X_data.keys():
            X_temp = Variable(torch.from_numpy(
                self.X_data[X_id]).float(), requires_grad=False)
            if self.is_gpu:
                self.X_data_var[X_id] = X_temp.cuda()
            else:
                self.X_data_var[X_id] = X_temp
        print("#")
        print("__input_transformation - end")
