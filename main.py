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
from dataloader import TCGADataset
from sklearn.model_selection import train_test_split

cuda = True if torch.cuda.is_available() else False

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot


def load_data(dataset,data_path,modalities, n_bins=4, eps: float = 1e-6) -> tuple:
    def split_omics(df):
        substrings = ['rnaseq', 'cnv', 'mut']
        dfs = [df.filter(like=sub) for sub in substrings]

        return {"omic":df, "rna-sequence":dfs[0],"mutation":dfs[2],"copy-number":dfs[1]}

    def filter_modalities(dict,modalities):
        features = []
        for key in dict.keys():
            if key in modalities:
                tensor_mod = torch.FloatTensor(dict[key].values)
                if cuda:
                    tensor_mod = tensor_mod.cuda()
                features.append(tensor_mod)
        return features

    test_size = 0.15
    val_size = 0.15

    data_path = Path(data_path).joinpath(f"omic/tcga_{dataset}_all_clean.csv.zip")
    df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)

    # handle missing values
    num_nans = df.isna().sum().sum()
    nan_counts = df.isna().sum()[df.isna().sum() > 0]
    df = df.fillna(df.mean(numeric_only=True))
    print(f"Filled {num_nans} missing values with mean")
    print(f"Missing values per feature: \n {nan_counts}")
    subset_df = df[df["censorship"] == 0] # only uncensored people

    label_col = "survival_months"
    disc_labels, q_bins = pd.qcut(subset_df[label_col], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = df[label_col].max() + eps
    q_bins[0] = df[label_col].min() - eps
    # use bin cuts to discretize all patients
    df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False,
                          include_lowest=True).values
    df["y_disc"] = df["y_disc"].astype(int)
    y = df["y_disc"].values
    omic_df = df.drop(
        ["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(omic_df, y, test_size = test_size, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    ######## implement dataset basen on tabular features
    X_train = filter_modalities(split_omics(X_train),modalities)
    X_val = filter_modalities(split_omics(X_val),modalities)
    X_test = filter_modalities(split_omics(X_test),modalities)


    print(f"Train samples: {int(len(y_train))}, Val samples: {int(len(y_test))}, "
          f"Test samples: {int(len(y_test))}")

    return (X_train,y_train), (X_val,y_val), (X_test,y_test)


def train_epoch(data_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss, _ = model(data_list, label)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()


def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)
    print(f'Model checkpoint saved to {filename}')


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def train(data_path='/net/archive/export/tcga/tcga',dataset='brca', modalites=  ["rna-sequence", "mutation", "copy-number"], testonly=False):
    test_inverval = 50
    if 'brca' in dataset:
        hidden_dim = [500]
        num_epoch = 2500
        lr = 1e-4
        step_size = 500
        num_class = 5
    elif 'ROSMAP' in dataset:
        hidden_dim = [300]
        num_epoch = 1000
        lr = 1e-4
        step_size = 500
        num_class = 2
    (X_train,y_train), (X_val,y_val), (X_test,y_test) = load_data(dataset,data_path,modalities=modalites)
    #data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder)
    labels_tr_tensor = torch.LongTensor(y_train)
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in X_train]
    model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    if not testonly:
        print("\nTraining...")
        for epoch in range(num_epoch+1):
            train_epoch(X_train, labels_tr_tensor, model, optimizer)
            scheduler.step()
            if epoch % test_inverval == 0:
                te_prob = test_epoch(X_val, model)
                print("\nVal: Epoch {:d}".format(epoch))
                if num_class == 2:
                    print("Val ACC: {:.5f}".format(accuracy_score(y_val, te_prob.argmax(1))))
                    print("Val F1: {:.5f}".format(f1_score(y_val, te_prob.argmax(1))))
                    print("Val AUC: {:.5f}".format(roc_auc_score(y_val, te_prob[:,1])))
                else:
                    print("Val ACC: {:.5f}".format(accuracy_score(y_val, te_prob.argmax(1))))
                    print("Val F1 weighted: {:.5f}".format(f1_score(y_val, te_prob.argmax(1), average='weighted')))
                    print("Val F1 macro: {:.5f}".format(f1_score(y_val, te_prob.argmax(1), average='macro')))
        save_checkpoint(model.state_dict(), modelpath)
    # test
    load_checkpoint(model, os.path.join(modelpath, 'checkpoint.pt'))
    te_prob = test_epoch(X_test, model)
    if num_class == 2:
        print("Test ACC: {:.5f}".format(accuracy_score(y_test, te_prob.argmax(1))))
        print("Test F1: {:.5f}".format(f1_score(y_test, te_prob.argmax(1))))
        print("Test AUC: {:.5f}".format(roc_auc_score(y_test, te_prob[:, 1])))
    else:
        print("Test ACC: {:.5f}".format(accuracy_score(y_test, te_prob.argmax(1))))
        print("Test F1 weighted: {:.5f}".format(f1_score(y_test, te_prob.argmax(1), average='weighted')))
        print("Test F1 macro: {:.5f}".format(f1_score(y_test, te_prob.argmax(1), average='macro')))



if __name__ == "__main__":
    data_folder = '/home/lw754/R255/data/BRCA'
    testonly = False
    modelpath = '/home/lw754/R255/mmdynamics/weights'
    train(data_path='/net/archive/export/tcga/tcga',dataset='brca', modalites=[ "mutation", "copy-number"], testonly=False)