""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, out_dim):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjusted input size based on output dimensions
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Middle
        x2 = self.encoder2(x1)

        x3 = self.middle(x2)
        x3 = F.interpolate(x3, size=(16, 16), mode='nearest')

        # Decoder
        x4 = self.decoder(torch.cat([x1, x3], dim=1))

        x4 = F.interpolate(x4, size=(64, 64), mode='nearest')
        return x4

class MMDynamic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout=0.5,class_weights=None,classification_loss_weight=1,feature_import_loss=1,modality_import_loss=1):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout  # probability
        self.class_weights = class_weights
        self.classification_loss_weight=classification_loss_weight
        self.feature_import_loss = feature_import_loss
        self.modality_import_loss = modality_import_loss
        assert self.views >= len(hidden_dim), f"to many hidden dimensions only {self.views} modalities given, but {len(hidden_dim)} hidden dims"
        len(hidden_dim)
        if self.views != len(hidden_dim):
            hidden_dim = self.views * hidden_dim



        # self.FeatureInforEncoder = nn.ModuleList( [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.FeatureInforEncoder = nn.ModuleList()
        for view in range(self.views):
            if isinstance(in_dim[view], int):
                self.FeatureInforEncoder.append(LinearLayer(in_dim[view], in_dim[view]))
            else:
                self.FeatureInforEncoder.append(UNet(in_channels=3, out_channels=1))

        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[i], 1) for i in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[i], num_class) for i in range(self.views)])
        #self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[view]) for view in range(self.views)])
        self.FeatureEncoder = nn.ModuleList()
        for view in range(self.views):
            if isinstance(in_dim[view], int):
                self.FeatureEncoder.append(LinearLayer(in_dim[view], hidden_dim[view]))
            else:
                self.FeatureEncoder.append(SmallCNN(hidden_dim[view]))


        if self.feature_import_loss is None:
            print('Feature importance is not used!')
        if self.modality_import_loss is None:
            print('Modality importance is not used!')


        ''' 
        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(np.sum(hidden_dim), hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(np.sum(hidden_dim), num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        '''
        self.MMClasifier = []
        self.MMClasifier.append(LinearLayer(np.sum(hidden_dim), num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False, tcp = False,feature_importance=False): # data_list just list with tensors (as many as modalities)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights,reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(
                self.FeatureInforEncoder[view](data_list[view]))  # extracts the importance for each feature
            if self.feature_import_loss is None:
                feature[view] = data_list[view]   # dont use feature importance
            else:
                feature[view] = data_list[view] * FeatureInfo[view]  # importance mulitplied by features
            feature[view] = self.FeatureEncoder[view](feature[view])  # reduce dim to 500
            feature[view] = F.relu(feature[view])  # relu activation function
            feature[view] = F.dropout(feature[view], self.dropout,
                                          training=self.training)  # applies dropout during training
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])  # calculate logits
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])  # estimate TCP
            if self.modality_import_loss is None:
                feature[view] = feature[view]                 # don't weight features corresponding to estimated TCP
            else:
                feature[view] = feature[view] * TCPConfidence[view] #  weight features corresponding to estimated TCP
        if feature_importance:
            return FeatureInfo
        MMfeature = torch.cat([i for i in feature.values()], dim=1)  # concatinate all featues
        MMlogit = self.MMClasifier(MMfeature)  # calculates final logits
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))  # final classifier loss
        tcps = []
        pred_tcps = []
        for view in range(self.views):
            if self.feature_import_loss is None:
                MMLoss = self.classification_loss_weight*MMLoss   # add l1 loss for feature importance
            else:
                MMLoss = self.classification_loss_weight * MMLoss + self.feature_import_loss * torch.mean(
                    FeatureInfo[view])  # add l1 loss for feature importance

            pred = F.softmax(TCPLogit[view], dim=1)  # probabilities for one modality
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(
                -1)  # unsqueezed tensor with all TCP for one modality for all samples
            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view],
                                                                                                        label))  # TCPConfidence[view].view(-1) unsqueeze as well makes list out of estimated TCP
            # criterion(TCPLogit[view], label) logits as vector and label as single int???
            if tcp:
                tcps.append(p_target)
                pred_tcps.append(TCPConfidence[view].view(-1))

            if self.modality_import_loss is None:
                MMLoss = MMLoss
            else:
                MMLoss = MMLoss + self.modality_import_loss * confidence_loss  # add confidence loss as well
        if tcp:
            return pred_tcps, tcps
        return MMLoss, MMlogit


    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

    def get_tcp(self, data_list,label):
        """
        :param data_list:
        :return: (estimated TCP, real TCP)
        """
        estimated_TCP, real_TCP = self.forward(data_list, label=label,tcp=True)
        return estimated_TCP,real_TCP

    def get_feature_importance(self,data_list):
        feature_views = self.forward(data_list, feature_importance=True)
        return feature_views


class MMStatic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout=0.5,class_weights=None,modality_weights=[1]):
        classification_loss_weight = 1
        feature_import_loss = None
        modality_import_loss = None
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout  # probability
        self.class_weights = class_weights
        self.classification_loss_weight=classification_loss_weight
        self.feature_import_loss = feature_import_loss
        self.modality_import_loss = modality_import_loss
        assert self.views >= len(hidden_dim), f"to many hidden dimensions only {self.views} modalities given, but {len(hidden_dim)} hidden dims"
        len(hidden_dim)
        if self.views != len(hidden_dim):
            hidden_dim = self.views * hidden_dim
        if self.views != len(modality_weights):
            modality_weights = self.views * modality_weights
        self.modality_weights = modality_weights

        self.FeatureEncoder = nn.ModuleList()
        for view in range(self.views):
            if isinstance(in_dim[view], int):
                self.FeatureEncoder.append(LinearLayer(in_dim[view], hidden_dim[view]))
            else:
                self.FeatureEncoder.append(SmallCNN(hidden_dim[view]))

        self.MMClasifier = []
        self.MMClasifier.append(LinearLayer(np.sum(hidden_dim), num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False, tcp = False,feature_importance=False): # data_list just list with tensors (as many as modalities)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights,reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            feature[view] = data_list[view]   # dont use feature importance
            feature[view] = self.FeatureEncoder[view](feature[view])  # reduce dim to 500
            feature[view] = F.relu(feature[view])  # relu activation function
            feature[view] = F.dropout(feature[view], self.dropout,
                                          training=self.training)  # applies dropout during training

        MMfeature = torch.cat([self.modality_weights[i]* latent_representation for i, latent_representation in enumerate(feature.values())], dim=1)  # concatinate all featues
        MMlogit = self.MMClasifier(MMfeature)  # calculates final logits
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))  # final classifier loss
        return MMLoss, MMlogit


    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


