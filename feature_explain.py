""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import MMDynamic
import pandas as pd
from pathlib import Path
from dataloader_2 import TCGADataset
from sklearn.model_selection import train_test_split
import dataloader_single
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import wandb
import argparse
import random
from mmdynamics_main import load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval the model")
    parser.add_argument("--modelpath", type=str, default="/home/lw754/R255/results/singlecell/mmdynamics/RUN_weighted_hiddim[35, 250]_num_epochs100_best/weights/checkpoint_31800.pt",
                        help="Results directory path")
    parser.add_argument("--root_folder", type=str, default="/home/lw754/R255/data/singlecell", help="Root folder path")
    use_wandb = True

    args = parser.parse_args()
    root_folder = Path(args.root_folder)

    donor = 13176
    df_meta = pd.read_csv(root_folder / 'meta_data_train.csv')
    df_protein = pd.read_csv(root_folder / 'protein_data_train.csv')
    df_rna = pd.read_csv(root_folder / 'rna_rand_data_train.csv')

    donors = np.unique(df_meta['donor'].values)
    relative_thresholds = []
    for donor in donors:
        df_dict, dim_list,class_weights = dataloader_single.prepare_data(df_meta, [df_protein,df_rna], donor=donor, batch_size=512)
        train_dataloader, val_dataloader, test_dataloader = df_dict['train'], df_dict['val'], df_dict['test']

        if use_wandb:
            wandb.init(project='R255 MMdynamics', name='Explainablity_Feature_importance', config={'model':args.modelpath},
                       resume="allow")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for training.")

        model = MMDynamic(dim_list, [35,250], 4)
        model.to(device)
        load_checkpoint(model, args.modelpath)
        model.eval()

        views = {'protein':[],'rna':[]}
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                features, label = data
                features = [modality.float().to(device) for modality in features]
                label = label.to(device)
                feature_views = model.get_feature_importance(features)
                views['protein'].extend(feature_views[0])
                views['rna'].extend(feature_views[1])

        views['protein'] = [tensor.detach().cpu().numpy() for tensor in views['protein']]
        views['rna'] = [tensor.detach().cpu().numpy() for tensor in views['rna']]

        print('#### PROTEIN ####')
        print(np.mean(views['protein'], axis=0))
        feature_import_protein = pd.DataFrame(np.mean(views['protein'], axis=0))
        feature_import_protein = feature_import_protein.T
        feature_import_protein.columns = list(df_protein.drop('cell_id', axis=1).columns)
        first_row = feature_import_protein.iloc[0]

        # Sort the values in descending order and get the top 5 column names
        top_columns = first_row.sort_values(ascending=False).head(10)
        last_columns = first_row.sort_values(ascending=True).head(10)
        print(last_columns)
        print(top_columns)

        print('#### RNA ####')
        print( np.mean(views['rna'], axis=0))
        feature_import_rna = pd.DataFrame(np.mean(views['rna'], axis=0))
        feature_import_rna = feature_import_rna.T
        feature_import_rna.columns = list(df_rna.drop('cell_id', axis=1).columns)
        first_row = feature_import_rna.iloc[0]

        # Sort the values in descending order and get the top 5 column names
        top_columns = first_row.sort_values(ascending=False).head(10)
        last_columns = first_row.sort_values(ascending=True).head(10)
        print(last_columns)
        print(top_columns)











