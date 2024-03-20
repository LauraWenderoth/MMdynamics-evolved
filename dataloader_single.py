import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split
import itertools
import random
from PIL import Image

class SingleCellDataset(Dataset):
    def __init__(self, X,y,image_path=None,transform=False):
        self.X = X
        self.y = y
        self.image_path = image_path
        if image_path is not None:
            self.image_path = Path(image_path)
        self.cuda = True if torch.cuda.is_available() else False


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = torch.tensor(self.y[idx])
        if self.image_path is None:
            features = [torch.tensor(modality.iloc[idx].values) for modality in self.X]
        else:
            features = []
            for modality in self.X:
                feature = modality.iloc[idx].values
                if len(feature) == 1:
                    image = Image.open(self.image_path/feature.item())
                    transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with a probability of 0.5
                        transforms.RandomVerticalFlip(p=0.5),
                    ])
                    tensor_transform = transforms.Compose([
                        transforms.Resize((64, 64)),
                        transforms.ToTensor()
                    ])
                    transformed_image = transform(image)
                    features.append(tensor_transform(transformed_image))
                else:
                    features.append(torch.tensor(feature))

        return features, label

def prepare_data(meta_df,dfs,donor = 31800,columns_to_drop = ['cell_id', 'day', 'donor', 'technology'],target='cell_type',dict_classes = {'BP': 0, 'EryP': 1, 'MoP': 2, 'NeuP': 3},batch_size=512,image_path=None):
    def make_train_test(df,ids,columns_to_drop,target):
        split = df[df['cell_id'].isin(ids)]
        split = split.copy()
        split.drop(columns=columns_to_drop, inplace=True)
        y = split[target].map(dict_classes).values
        X = split.drop(columns=[target], inplace=False)
        return X,y



    merged_dfs = []
    test = meta_df[meta_df['donor'] == donor].copy()
    train_pre = meta_df[meta_df['donor'] != donor].copy()
    test_ids = test['cell_id'].values
    train_ids, val_ids = train_test_split(train_pre['cell_id'].values, test_size=0.2,
                                           random_state=42)  # You can adjust the test_size as needed
    for df in dfs:
        merged_df = pd.merge(meta_df, df, on='cell_id', how='inner')
        merged_df = merged_df.drop_duplicates(subset='cell_id', keep='first')
        merged_dfs.append(merged_df)
    dfs = [[],[],[]]

    for df in merged_dfs:
        X_train,y_train = make_train_test(df, train_ids, columns_to_drop, target)
        X_val, y_val = make_train_test(df, val_ids, columns_to_drop, target)
        X_test, y_test = make_train_test(df, test_ids, columns_to_drop, target)
        dfs[0].append(X_train)
        dfs[1].append(X_val)
        dfs[2].append(X_test)

    class_counts = torch.bincount(torch.tensor(y_train))
    total_samples = class_counts.sum().float()
    class_frequencies = class_counts / total_samples
    class_weights = 1.0 / class_frequencies


    dim_list = [x.shape[1] for x in dfs[0]]
    dim_list = [(3, 64, 64) if 1 == dim else dim for dim in dim_list]

    train_set = SingleCellDataset(dfs[0], y_train, image_path=image_path)
    val_set = SingleCellDataset(dfs[1], y_val, image_path=image_path)
    test_set = SingleCellDataset(dfs[2], y_test, image_path=image_path)

    train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=True, num_workers=0)

    dataloader = {'train': train_dataloader, 'val': val_dataloader,'test': test_dataloader}

    return dataloader,dim_list, class_weights
