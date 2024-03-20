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
    use_wandb = False

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
            wandb.init(project='R255 MMdynamics', name='Explainablity_TCP', config={'model':args.modelpath},
                       resume="allow")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for training.")

        model = MMDynamic(dim_list, [35,250], 4)
        model.to(device)
        load_checkpoint(model, args.modelpath)
        model.eval()

        real_TCPs = []
        estimated_TCPs = []
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                features, label = data
                features = [modality.float().to(device) for modality in features]
                label = label.to(device)
                estimated_TCP, real_TCP = model.get_tcp(features,label)
                real_TCPs.extend(real_TCP[1])
                estimated_TCPs.extend(estimated_TCP[1])

        y_true = np.squeeze([tensor.detach().cpu().numpy() for tensor in real_TCPs])
        y_pred = np.squeeze([tensor.detach().cpu().numpy() for tensor in estimated_TCPs])
        absolute_errors = np.abs(np.subtract(y_true, y_pred))
        threshold = np.array(y_true) * 0.1
        num_errors_bigger_than_threshold = np.sum(absolute_errors > threshold)
        proportion_errors = num_errors_bigger_than_threshold / len(y_true)

        # Print the proportion of errors relative to the total number of samples

        print("Mean Squared Error:", mean_squared_error(y_true, y_pred))
        print("Root Mean Squared Error:", mean_squared_error(y_true, y_pred, squared=False))  # square=False for RMSE
        print("Mean Absolute Error:", mean_absolute_error(y_true, y_pred))
        print("Standard Deviation of Absolute Error:", np.std(absolute_errors))
        print("Maximum Absolute Error:", np.max(absolute_errors))
        print("Minimum Absolute Error:", np.min(absolute_errors))
        print("Coefficient of Determination (R-squared):", r2_score(y_true, y_pred))
        print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_true, y_pred))
        #print("Number of errors bigger than 10% of original true values:", num_errors_bigger_than_threshold)
        #print("Proportion of errors bigger than 10% of original true values:", proportion_errors)
        relative_threshold = []
        for i in range(1,11):
            i = i/10
            threshold = np.array(y_true) * i
            num_errors_bigger_than_threshold = np.sum(absolute_errors > threshold)
            proportion_errors = num_errors_bigger_than_threshold / len(y_true)
            relative_threshold.append(proportion_errors)
            print(f"Number of errors bigger than {i*100}% of original true values:", num_errors_bigger_than_threshold)
            print(f"Proportion of errors bigger than {i*100}% of original true values:", proportion_errors)
        relative_thresholds.append(relative_threshold)
        if use_wandb:
            mse = mean_squared_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            std_absolute_errors = np.std(absolute_errors)
            max_absolute_error = np.max(absolute_errors)
            min_absolute_error = np.min(absolute_errors)
            r_squared = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            # Log metrics
            wandb.log({
                "Mean Squared Error": mse,
                "Root Mean Squared Error": rmse,
                "Mean Absolute Error": mae,
                "Standard Deviation of Absolute Error": std_absolute_errors,
                "Maximum Absolute Error": max_absolute_error,
                "Minimum Absolute Error": min_absolute_error,
                "Coefficient of Determination (R-squared)": r_squared,
                "Mean Absolute Percentage Error": mape
            })

            # Log proportion of errors bigger than threshold
            for i in range(1, 11):
                threshold = i / 10 * np.array(y_true)
                num_errors_bigger_than_threshold = np.sum(absolute_errors > threshold)
                proportion_errors = num_errors_bigger_than_threshold / len(y_true)
                wandb.log({
                    f"Number of errors bigger than {i * 10}% of original true values": num_errors_bigger_than_threshold,
                    f"Proportion of errors bigger than {i * 10}% of original true values": proportion_errors
                })

            # Finish logging
    means = np.mean(relative_thresholds,axis=0)
    stds = np.std(relative_thresholds,axis=0)
    mins = np.min(relative_thresholds, axis=0)
    maxs = np.max(relative_thresholds, axis=0)
    print(means)
    print(stds)
    print(mins)
    print(maxs)
    if use_wandb:
        wandb.finish()




