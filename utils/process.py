import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from contextualized.easy import ContextualizedRegressor,ContextualizedCorrelationNetworks

#ar = np.load("/home/xuxi/Ruize_projects/data/data_files.npz", allow_pickle=True)
if __name__ == '__main__':
    
    ar = np.load("../data/data_files.npz", allow_pickle=True)
    print(ar)
    epi = ar['atac_train_small']
    rna = ar['rna_train']
    cell_types = ar['cell_type_train']
    ct_list = sorted(list(set(cell_types)))

    epi_pca = PCA(n_components=5)
    epi_small = epi_pca.fit_transform(epi)
    rna_pca = PCA(n_components=10)
    rna_small = rna_pca.fit_transform(rna)

    # Pre-process and normalize data.
    X = rna_small.astype(np.float32)
    C = epi_small.astype(np.float32)
    C = np.hstack((C, X[:, :2]))

    # Get train-test split
    train_idx, test_idx = train_test_split(range(len(X)), test_size=0.25, random_state=1)
    X_train, X_test = X[train_idx], X[test_idx]
    C_train, C_test = C[train_idx], C[test_idx]
    cell_types_train, cell_types_test = cell_types[train_idx], cell_types[test_idx]
    rna_train, rna_test = rna[train_idx], rna[test_idx]
    epi_train, epi_test = epi[train_idx], epi[test_idx]

    # Normalize the data
    C_means = np.mean(C_train, axis=0)
    C_stds = np.std(C_train, axis=0)
    C_train -= C_means
    C_test -= C_means
    C_train /= C_stds
    C_test /= C_stds

    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0)
    X_train -= X_means
    X_test -= X_means
    X_train /= X_stds
    X_test /= X_stds

    
    # model = ContextualizedRegressor()
    # model.fit(C, X, Y)
    model=ContextualizedCorrelationNetworks(encoder_type='ngam',num_archetypes=16,n_bootstrap=3)
    model.fit(C,X,max_epochs=5)
    # Get rho
    rho = model.predict_correlation(C, individual_preds=False, squared=False)

    # Get rho^2
    rho_squared = model.predict_correlation(C, individual_preds=False, squared=True)
    print(rho)
    print(rho_squared)