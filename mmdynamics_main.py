""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import MMDynamic, MMStatic
import pandas as pd
from pathlib import Path
from dataloader_2 import TCGADataset
from sklearn.model_selection import train_test_split
import dataloader_single
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
import wandb
import argparse
import random
from models import HealNet, MCATomics

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot





def train_epoch(train_dataloader, model, optimizer,device):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_dataloader):
        features, labels = data
        features = [modality.float().to(device) for modality in features]
        labels = labels.to(device)
        optimizer.zero_grad()
        loss, _ = model(features, labels)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return model, epoch_loss


    # print(f'Running loss: {epoch_loss/len(train_dataloader)}')


def eval_model(dataloader, model,device,title='Val'):
    model.eval()
    probas = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            features, label = data
            features = [modality.float().to(device) for modality in features]
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
    results = {f'{title}_f1':f1,f'{title}_f1_macro':f1_macro,f'{title}_recall':recall,f'{title}_precision':precision,f'{title}_accuracy':accuracy,f'{title}_balanced_accuracy':balanced_accuracy}
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
    for model, result_data in models_dict.items():
        row_data = [model]
        for metric, value in result_data.items():
            row_data.append(value)
        data.append(row_data)

    df_results = pd.DataFrame(data, columns=[ 'Model','f1', 'f1_macro', 'recall', 'precision', 'accuracy',
                                             'balanced_accuracy','Donor'])
    df_results.set_index('Model', inplace=True)


    return df_results



def train(root_folder,results_dir,device, hidden_dim =[70, 500], num_epochs = 100, modalities=['protein','rna'], batch_size=1024, testonly=False, use_wandb=True,model_name='checkpoint.pt',classes = {'BP': 0, 'EryP': 1, 'MoP': 2, 'NeuP': 3},classification_loss_weight=1,feature_import_loss=1,modality_import_loss=1,image_dir=None, model_to_load = 'mmdynamics',modality_weights=[1],model_weights=None):

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    class_names = list(classes.keys())
    test_inverval = 10
    lr = 1e-4
    step_size = 500
    num_class = 4

    run_name = f'RUN_weighted_{modalities}_hiddim{hidden_dim}_num_epochs{num_epochs}_best_{model_to_load}_{modality_weights}_{classification_loss_weight}_{feature_import_loss}_{modality_import_loss}'

    results_dir = results_dir / run_name
    results_dir.mkdir(exist_ok=True)

    modelpath = results_dir / 'weights'
    modelpath.mkdir(exist_ok=True)

    if use_wandb:
        wandb.init(project='R255 MMdynamics',name=run_name, config={"hidden_dim": hidden_dim, "num_epochs": num_epochs,
                                                                    "lr": lr, "save_dir" :results_dir,
                                                                    'classification_loss_weight':classification_loss_weight,
                                                                    'feature_import_loss':feature_import_loss,
                                                                    'modality_import_loss':modality_import_loss},
                   resume="allow")
    ###
    df_meta = pd.read_csv(root_folder / 'meta_data_train.csv')
    df_protein = pd.read_csv(root_folder / 'protein_data_train.csv')
    df_rna = pd.read_csv(root_folder / 'rna_rand_data_train.csv')
    df_images = pd.read_csv(root_folder / 'image_data.csv')

    donors = np.unique(df_meta['donor'].values)
    donor_dfs =[]
    used_modalities = []
    if 'protein' in modalities:
        used_modalities.append(df_protein)
    if 'rna' in modalities:
        used_modalities.append(df_rna)
    if 'image' in modalities:
        used_modalities.append(df_images)

    print(f'Used modalities: {modalities}')
    print(f'Loss weighting: classification loss {classification_loss_weight}, feature importance {feature_import_loss}, modality importance {modality_import_loss}')


    for donor in donors:
        df = dataloader_single.prepare_data(df_meta, used_modalities, donor=donor,batch_size=batch_size,image_path=image_dir)
        donor_dfs.append(df)

    results = []
    cmfs = []
    for donor_i,(df_dict, dim_list,class_weights) in enumerate(donor_dfs):
        class_weights = class_weights.to(device)
        print('#################')
        train_dataloader, val_dataloader, test_dataloader = df_dict['train'], df_dict['val'], df_dict['test']
        if model_to_load == 'mmdynamics':
            model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5, class_weights=class_weights,classification_loss_weight=classification_loss_weight,feature_import_loss=feature_import_loss,modality_import_loss=modality_import_loss)
            print('Using MMdynamic as model')
        elif model_to_load == 'mmstatic':
            model = MMStatic(dim_list, hidden_dim, num_class, dropout=0.5,class_weights=class_weights,modality_weights=modality_weights)
            print('Using MMstatic as model')
        elif model_to_load == 'healnet':
            model = HealNet(
                modalities=len(modalities),
                input_channels=dim_list,  # number of features as input channels
                input_axes=[1]*len(modalities),  # second axis (b n_feats c)
                num_classes=num_class,
                num_freq_bands=2,
                depth=2,
                max_freq=2.0,
                num_latents=17,
                latent_dim=126,
                cross_dim_head=63,
                latent_dim_head=20,
                cross_heads=1,
                latent_heads=8,
                attn_dropout=0.5,
                ff_dropout=0.36,
                weight_tie_layers=False,
                fourier_encode_data=True,
                self_per_cross_attn=0,
                final_classifier_head=True,
                snn=True,
                class_weights=class_weights,
            )
            print('Using Healnet as model')
        elif model_to_load == 'mcat':
            assert len(modalities) <= 2, f'MCAT only for 2 modalities but given {len(modalities)} modalities'
            model = MCATomics(
                n_classes=num_class,
                omic_shape1=torch.Size([dim_list[0]]),
                omic_shape2=torch.Size([dim_list[1]]) ,
                class_weights=class_weights,
            )
            print('Using MCAT as model')
        else:
            print('Choose a valid option for your model type!')
        model.to(device)
        best_model = model
        best_f1_macro = 0
        if not testonly:

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
            print("\nTraining...")
            for epoch in tqdm(range(1,num_epochs + 1), desc="Epochs"):
                model, loss = train_epoch(train_dataloader, model, optimizer,device=device)
                if use_wandb:
                    wandb.log({"loss": loss / len(train_dataloader), "epoch": epoch})
                scheduler.step()
                if epoch % test_inverval == 0:
                    result,cfm = eval_model(val_dataloader, model,device)
                    result["epoch"] = epoch
                    if use_wandb:
                        wandb.log(result)
                    if best_f1_macro < result['Val_f1_macro']:
                        best_f1_macro = result['Val_f1_macro']
                        best_model = model
            model_name = f'checkpoint_{donors[donor_i]}.pt'
            save_checkpoint(best_model.state_dict(), modelpath,filename=model_name)
        # test
        if not testonly:
            load_checkpoint(best_model, os.path.join(modelpath,model_name))
        else:
            model_name = f'checkpoint_{donors[donor_i]}.pt'
            load_checkpoint(best_model, os.path.join(model_weights, model_name))
        result,cfm = eval_model(test_dataloader, model,device=device,title='Test')
        result['Donor'] = donors[donor_i]

        results.append(transform_to_df({'MM dynamics': result}))
        cmfs.append(cfm)
        if use_wandb:
            wandb.log(result)
            wandb.log({f'Confusions_matrix_{donors[donor_i]}': wandb.Table(data=cfm, columns=class_names, rows=class_names)})
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
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")

    # Add arguments
    parser.add_argument("--root_folder", type=str, default="/home/lw754/R255/data/singlecell", help="Root folder path")
    parser.add_argument("--results_dir", type=str, default="/home/lw754/R255/results/singlecell/mmdynamics", help="Results directory path")
    parser.add_argument("--image_dir", type=str, default="/home/lw754/R255/data/bm_images", help="Root folder to images")
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[35,250,500], help="Hidden dimensions")
    parser.add_argument("--modalities", nargs="+", type=str, default=['protein', 'rna','image'], help="Modalities")
    parser.add_argument("--testonly", action="store_true", default=True, help="Test only mode")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
    parser.add_argument("--classification_loss_weight", type=float, default=1, help="Classification loss weight")
    parser.add_argument("--feature_import_loss", type=float, default=1, help="Feature import loss weight")
    parser.add_argument("--modality_import_loss", type=float, default=1, help="Modality import loss weight")
    parser.add_argument("--model", type=str, default='mmstatic', help="mmdynamics, mmstatic,healnet,mcat")
    parser.add_argument("--modality_weights", nargs="+", type=int, default=[3,1,6], help="weight assigned to each latent representation before concartination")
    parser.add_argument("--model_weights", type=str,
                        #default='/home/lw754/R255/results/singlecell/mmdynamics/RUN_weighted_hiddim[35, 250, 500]_num_epochs250_best_1_1_1/weights/',
                        default="/home/lw754/R255/results/singlecell/mmdynamics/RUN_weighted_['protein', 'rna', 'image']_hiddim[35, 250, 500]_num_epochs250_best_mmstatic_[3, 1, 6]_1_1_1/weights/",
                        help="mmdynamics, mmstatic,healnet,mcat")


    # Parse arguments
    args = parser.parse_args()

    # Convert paths to Path objects
    root_folder = Path(args.root_folder)
    results_dir = Path(args.results_dir)
    image_dir = Path(args.image_dir)

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for training.")
    # Call the train function
    train(root_folder=root_folder, results_dir=results_dir, device=device, hidden_dim=args.hidden_dim, modalities=args.modalities, testonly=args.testonly,
          num_epochs = args.epochs,use_wandb=False,classification_loss_weight=args.classification_loss_weight,feature_import_loss=args.feature_import_loss,modality_import_loss=args.modality_import_loss,image_dir=image_dir,model_to_load=args.model,modality_weights=args.modality_weights,model_weights=args.model_weights)