import argparse
from email.policy import default
from tkinter import E
from xmlrpc.client import boolean
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import os

from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_PI, optimizer_EI, optimizer_UCB, max_PI, max_EI, max_UCB
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

import pickle



def train_loop(model:str, feature_path:Path, target_path:Path, al_isTrue:boolean):
    if model == 'GaussianProcesses' or model=='GP':
        train_x = dd.read_parquet(feature_path)
        train_y= dd.read_parquet(target_path)

        optimizer = BayesianOptimizer(
            estimator=GPR(kernel=Matern(length_scale=1.0)),
            X_training=train_x, y_training=train_y,
            query_strategy=max_EI
        )
        
        n_queries = 1000
        for n_query in range(n_queries):

            query_idx,query_inst = optimizer.query(train_x)
            optimizer.teach(train_x[query_idx].reshape(1,-1), train_y[query_inst].reshape(1,-1), only_new=True)
        
        with open(os.path.dirname(os.getcwd()) + '/models/GPR/{}_{}'.format(model,str(al_isTrue))) as f:
            pickle.dump(optimizer.estimator,f)

def test_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_type','-at',
                        dest='access',
                        required=True,
                        choices = {'train','test'},
                        help='Specifying training and testing loops')

    parser.add_argument('--model','-m',
                        dest='model',
                        default='GaussianProcesses',
                        choices = {'GaussianProcesses','RandomForest','GP','RF'},
                        help='Specifying the type of model')

    parser.add_argument('--features','-f',
                        dest='features',
                        type = Path,
                        required=True,
                        help='X_values / Feature Matrix')

    parser.add_argument('--target','-t',
                        dest='target',
                        type = Path,
                        required=True,
                        help='Y_values / Target Matrix')

    parser.add_argument('--activeLearning','-al',
                        dest='al',
                        # default=True,
                        help='Whether to use active learning or not')

    args = parser.parse_args()

    if args.access == 'train':
        train_loop(args.model,args.features,args.target,args.al)
    elif args.access == 'test':
        test_loop(args.model,args.features,args.target,args.al)
