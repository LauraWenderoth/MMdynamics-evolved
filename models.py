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



        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[i], 1) for i in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[i], num_class) for i in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[view]) for view in range(self.views)])

        if self.feature_import_loss is None:
            print('Feature importance is not used!')
        if self.modality_import_loss is None:
            print('Modality importance is not used!')



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

    def forward(self, data_list, label=None, infer=False, tcp = False): # data_list just list with tensors (as many as modalities)
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

        MMfeature = torch.cat([i for i in feature.values()], dim=1)  # concatinate all featues
        MMlogit = self.MMClasifier(MMfeature)  # calculates final logits
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))  # final classifier loss
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
                return TCPConfidence[view].view(-1), p_target

            if self.modality_import_loss is None:
                MMLoss = MMLoss
            else:
                MMLoss = MMLoss + self.modality_import_loss * confidence_loss  # add confidence loss as well
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




