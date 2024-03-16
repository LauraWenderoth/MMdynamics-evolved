import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
class TCGADataset(Dataset):
    def __init__(self, features:list,labels,image_path,modalities, ids, n_bins=4, eps: float = 1e-6):
        self.image_path = image_path
        self.modalities = modalities
        self.cuda = True if torch.cuda.is_available() else False
        self.slide_ids = ids
        self.omic_features = features
        self.labels = labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.image_path, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        label = self.labels[idx]
        label = torch.LongTensor(label)
        features = [torch.FloatTensor(modality[idx]) for modality in self.omic_features]
        return features, label


def load_data(dataset,data_path,modalities, n_bins=4, batch_size=32, eps: float = 1e-6) -> tuple:
    def split_omics(df):
        substrings = ['rnaseq', 'cnv', 'mut']
        dfs = [df.filter(like=sub) for sub in substrings]

        return {"omic":df, "rna-sequence":dfs[0],"mutation":dfs[2],"copy-number":dfs[1]}

    def filter_modalities(dict,modalities):
        features = []
        for key in dict.keys():
            if key in modalities:
                features.append(dict[key].values)
        return features

    test_size = 0.15
    val_size = 0.15

    omic_path = Path(data_path).joinpath(f"omic/tcga_{dataset}_all_clean.csv.zip")
    df = pd.read_csv(omic_path, compression="zip", header=0, index_col=0, low_memory=False)

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
        ["site", "oncotree_code", "train","case_id",  "censorship", "survival_months", "y_disc"], axis=1) #,
    X_train, X_test, y_train, y_test = train_test_split(omic_df, y, test_size = test_size, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    ID_train = X_train[["slide_id"]].values
    ID_test = X_test[["slide_id"]].values
    ID_val = X_val[["slide_id"]].values

    X_train = X_train.drop(["slide_id"], axis=1)
    X_val = X_val.drop([ "slide_id"], axis=1)
    X_test = X_test.drop([ "slide_id"], axis=1)

    ######## implement dataset basen on tabular features
    X_train = filter_modalities(split_omics(X_train),modalities)
    X_val = filter_modalities(split_omics(X_val),modalities)
    X_test = filter_modalities(split_omics(X_test),modalities)


    print(f"Train samples: {int(len(y_train))}, Val samples: {int(len(y_test))}, "
          f"Test samples: {int(len(y_test))}")
    image_path = Path(data_path).joinpath(f"wsi/{dataset}")
    train_set = TCGADataset(X_train,y_train,image_path=image_path,modalities=modalities, ids=ID_train)
    val_set = TCGADataset(X_val, y_val, image_path=image_path, modalities=modalities, ids=ID_val)
    test_set = TCGADataset(X_test, y_test, image_path=image_path, modalities=modalities, ids=ID_test)

    train_dataloader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    os.chdir("../../")
    data_path = '/net/archive/export/tcga/tcga'
    dataset='brca'
    modalites = ["rna-sequence", "mutation", "copy-number"]
    load_data(dataset, data_path, modalities=modalites)
