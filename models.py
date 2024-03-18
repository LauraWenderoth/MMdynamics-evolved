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
    def __init__(self, in_dim, hidden_dim, num_class, dropout,class_weights=None):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout  # probability
        self.class_weights = class_weights
        if self.views != len(hidden_dim):
            hidden_dim = self.views * hidden_dim

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[i], 1) for i in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[i], num_class) for i in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[view]) for view in range(self.views)])



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

    def forward(self, data_list, label=None, infer=False): # data_list just list with tensors (as many as modalities)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights,reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(
                self.FeatureInforEncoder[view](data_list[view]))  # extracts the importance for each feature
            feature[view] = data_list[view] * FeatureInfo[view]  # importance mulitplied by features
            feature[view] = self.FeatureEncoder[view](feature[view])  # reduce dim to 500
            feature[view] = F.relu(feature[view])  # relu activation function
            feature[view] = F.dropout(feature[view], self.dropout,
                                      training=self.training)  # applies dropout during training
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])  # calculate logits
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])  # estimate TCP
            feature[view] = feature[view] * TCPConfidence[view]  # weight features corresponding to estimated TCP

        MMfeature = torch.cat([i for i in feature.values()], dim=1)  # concatinate all featues
        MMlogit = self.MMClasifier(MMfeature)  # calculates final logits
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))  # final classifier loss
        for view in range(self.views):
            MMLoss = MMLoss + torch.mean(FeatureInfo[view])  # add l1 loss for feature importance
            pred = F.softmax(TCPLogit[view], dim=1)  # probabilities for one modality
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(
                -1)  # unsqueezed tensor with all TCP for one modality for all samples
            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view],
                                                                                                        label))  # TCPConfidence[view].view(-1) unsqueeze as well makes list out of estimated TCP
            # criterion(TCPLogit[view], label) logits as vector and label as single int???
            MMLoss = MMLoss + confidence_loss  # add confidence loss as well
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit




