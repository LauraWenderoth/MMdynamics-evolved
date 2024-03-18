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
import wandb

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot





def train_epoch(train_dataloader, model, optimizer):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_dataloader):
        features, labels = data
        features = [modality.float().cuda() for modality in features]
        labels = labels.cuda()
        optimizer.zero_grad()
        loss, _ = model(features, labels)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # print(f'Running loss: {epoch_loss/len(train_dataloader)}')


def eval_model(dataloader, model,title='Val'):
    model.eval()
    probas = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            features, label = data
            features = [modality.float().cuda() for modality in features]
            logit = model.infer(features)
            prob = F.softmax(logit, dim=1).data.cpu().numpy()
            probas.extend(prob)
            y_true.extend(label)
    predictions = np.argmax(probas,axis=1)

    f1 = f1_score(y_true, predictions, average='weighted')
    f1_macro = f1_score(y_true, predictions, average='macro')
    recall = recall_score(y_true, predictions, average='weighted')
    precision = precision_score(y_true, predictions, average='weighted')
    accuracy = accuracy_score(y_true, predictions)
    balanced_accuracy = balanced_accuracy_score(y_true, predictions)
    print(f"Evaluation of {title} split: F1 Score: {f1},F1 Score macro: {f1_macro}, Recall: {recall}, Precision: {precision}, Accuracy: {accuracy}, balanced Acc {balanced_accuracy}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, predictions)
    results = {'f1':f1,'f1_macro':f1_macro,'recall':recall,'precision':precision,'accuracy':accuracy,'balanced_accuracy':balanced_accuracy}
    return results,conf_matrix

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)
    print(f'Model checkpoint saved to {filename}')


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)

def transform_to_df(models_dict):
    data = []
    # Iterate over each model and its corresponding results
    for model, result_data in models_dict.items():
        # Append the model name as the first element of the data row
        row_data = [model]
        # Append the evaluation metrics for the model
        for metric, value in result_data.items():
            row_data.append(value)
        # Append the row data to the main data list
        data.append(row_data)

    # Create a DataFrame from the data
    df_results = pd.DataFrame(data, columns=['Model', 'f1', 'f1_macro', 'recall', 'precision', 'accuracy',
                                             'balanced_accuracy'])
    df_results.set_index('Model', inplace=True)


    return df_results


def train(root_folder,results_dir, batch_size=1024, testonly=False, wandb=True):


    test_inverval = 100
    hidden_dim = [70, 500]
    num_epochs = 50 #2500
    lr = 1e-4
    step_size = 500
    num_class = 4

    project_name = f'RUN_weighted_hiddim{hidden_dim}_num_epochs{num_epochs}'

    results_dir = results_dir / project_name
    results_dir.mkdir(exist_ok=True)

    modelpath = results_dir / 'weights'
    modelpath.mkdir(exist_ok=True)

    if wandb:
        wandb.init(project=project_name, config={"hidden_dim": hidden_dim, "num_epochs": num_epochs, "lr": lr, "save_dir" :results_dir})
    ###
    df_meta = pd.read_csv(root_folder / 'meta_data_train.csv')
    df_protein = pd.read_csv(root_folder / 'protein_data_train.csv')
    df_rna = pd.read_csv(root_folder / 'rna_rand_data_train.csv')

    donors = np.unique(df_meta['donor'].values)
    donor_dfs =[]

    for donor in donors:
        df = dataloader_single.prepare_data(df_meta, [df_protein, df_rna], donor=donor,batch_size=batch_size)
        donor_dfs.append(df)

    results = []
    cmfs = []
    for (df_dict, dim_list,class_weights) in donor_dfs:
        class_weights = class_weights.cuda()
        print('#################')
        train_dataloader, val_dataloader, test_dataloader = df_dict['train'], df_dict['val'], df_dict['test']
        model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5, class_weights=class_weights)
        model.cuda()
        if not testonly:

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
            print("\nTraining...")
            for epoch in tqdm(range(1,num_epochs + 1), desc="Epochs"):
                train_epoch(train_dataloader, model, optimizer)
                scheduler.step()
                if epoch % test_inverval == 0:
                    result,cfm = eval_model(val_dataloader, model)

            save_checkpoint(model.state_dict(), modelpath)
        # test
        load_checkpoint(model, os.path.join(modelpath, 'checkpoint.pt'))
        result,cfm = eval_model(test_dataloader, model,title='Test')
        results.append(transform_to_df({'MM dynamics': result}))
        cmfs.append(cfm)
    # calc result
    mean_df = pd.concat(results).groupby(level=0).mean()
    std_df = pd.concat(results).groupby(level=0).std()
    print(mean_df.to_latex())
    print(std_df.to_latex())
    mean_df.to_csv(results_dir / f'metrics_mean.csv')
    std_df.to_csv(results_dir / f'metrics_std.csv')

    stacked_cfms = np.stack(cmfs, axis=0)
    mean_cfm = np.mean(stacked_cfms, axis=0)
    std_cfm = np.std(stacked_cfms, axis=0)
    np.savetxt( results_dir / f'cfm_mean.csv',mean_cfm,delimiter=',')
    np.savetxt(results_dir / f'cfm_std.csv', std_cfm, delimiter=',')

if __name__ == "__main__":
    root_folder = Path("/home/lw754/R255/data/singlecell")
    results_dir = Path("/home/lw754/R255/results/singlecell/mmdynamics")
    train(root_folder=root_folder, results_dir=results_dir,testonly=False)