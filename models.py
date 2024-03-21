""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, einsum
from typing import *
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from healnet.models.healnet import PreNorm, Attention, FeedForward, cache_fn, fourier_encode
from healnet.baselines.mcat import SNN_Block, BilinearFusion, Attn_Net_Gated, MultiheadAttention
from math import pi, log
from functools import wraps

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


class HealNet(nn.Module):
    def __init__(
        self,
        *,
        modalities: int,
        num_freq_bands: int = 2,
        depth: int = 3,
        max_freq: float=2,
        input_channels: List,
        input_axes: List,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 1000,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        snn: bool = True,
        class_weights = None
    ):
        super().__init__()
        assert len(input_channels) == len(input_axes), 'input channels and input axis must be of the same length'
        assert len(input_axes) == modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = input_axes
        self.input_channels=input_channels
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = modalities
        self.self_per_cross_attn = self_per_cross_attn
        self.class_weights = class_weights

        self.fourier_encode_data = fourier_encode_data

        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in input_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, input_channels):
            input_dims.append(f_channels + i_channels)


        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # modality-specific attention layers
        funcs = []
        for m in range(modalities):
            funcs.append(lambda m=m: PreNorm(latent_dim, Attention(latent_dim, input_dims[m], heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout, snn = snn))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])


        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))


            cross_attn_layers = []
            for j in range(modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))


            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()


    # def _handle_missing(self, tensors: List[torch.Tensor]):



    def forward(self,
                tensors: List[torch.Tensor],
                labels=None,
                mask: Optional[torch.Tensor] = None,
                missing: Optional[torch.Tensor] = None,
                return_embeddings: bool = False,
                infer: bool = False,
                ):

        for i in range(len(tensors)):
            data = tensors[i]
            data = torch.unsqueeze(data, dim=1)
            # sanity checks
            b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
            assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                          f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                pos = torch.linspace(0, 1, axis[0], device = device, dtype = dtype)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, 'n d -> () n d')
                enc_pos = repeat(enc_pos, '() n d -> b n d', b = b)
                data = torch.cat((data, enc_pos), dim = -1)

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d')
            tensors[i] = data


        x = repeat(self.latents, 'n d -> b n d', b = b) # note: batch dim should be identical across modalities

        for layer in self.layers:
            for i in range(self.modalities):
                cross_attn= layer[i*2]
                cross_ff = layer[(i*2)+1]
                try:
                    x = cross_attn(x, context = tensors[i], mask = mask) + x
                    x =  cross_ff(x) + x
                except:
                    pass

            if self.self_per_cross_attn > 0:
                self_attn, self_ff = layer[-1]

                x = self_attn(x) + x
                x = self_ff(x) + x

        if return_embeddings:
            return x
        logit = self.to_logits(x)
        if infer:
            return logit
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        loss = torch.mean(criterion(logit, labels))
        return loss, logit

    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Helper function which returns all attention weights for all attention layers in the model
        Returns:
            all_attn_weights: list of attention weights for each attention layer
        """
        all_attn_weights = []
        for module in self.modules():
            if isinstance(module, Attention):
                all_attn_weights.append(module.attn_weights)
        return all_attn_weights

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

class MCATomics(nn.Module):
    def __init__(self, omic_shape1: Tuple, omic_shape2: Tuple, fusion='concat', n_classes=4,
                 model_size_omic1: str='small', model_size_omic2: str='small', dropout=0.25,class_weights=None):
        """
        Multimodal Co-Attention Transformer
        Args:
            fusion:
            omic_shape:
            n_classes:
            model_size_wsi:
            model_size_omic:
            dropout:
        """
        super(MCATomics, self).__init__()
        self.fusion = fusion
        self.omic_shape1 = omic_shape1
        self.omic_shape2 = omic_shape2
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        self.class_weights = class_weights

        ### Constructing Genomic SNN
        hidden1 = self.size_dict_omic[model_size_omic1]
        sig_networks1 = []
        for input_dim in self.omic_shape1:
            fc_omic1 = [SNN_Block(dim1=input_dim, dim2=hidden1[0])]
            for i, _ in enumerate(hidden1[1:]):
                fc_omic1.append(SNN_Block(dim1=hidden1[i], dim2=hidden1[i + 1], dropout=0.25))
            sig_networks1.append(nn.Sequential(*fc_omic1))
        self.sig_networks1 = nn.ModuleList(sig_networks1)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic2]
        sig_networks = []
        for input_dim in self.omic_shape2:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks2 = nn.ModuleList(sig_networks)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        #size
        size = [1024, 256, 256]
        size[0] = omic_shape1[0]  # other embedding dims

        ### Omic Transformer1 + Attention Head
        omic_encoder_layer1 = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.omic_transformer1 = nn.TransformerEncoder(omic_encoder_layer1, num_layers=2)
        self.omic_attention_head1 = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho1 = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Omic Transformer2 + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        ### ClassifMCATier
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, data, labels = None,infer=False, **kwargs):
        # x_path = kwargs['x_path']
        # x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        x_omic1 = [data[0]] # needs to be wrapped in list (used to expect multiple omic sources)
        h_omic1 = [self.sig_networks1[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic1)] ### each omic signature goes through it's own FC layer

        # h_omic_bag = torch.stack(h_omic).unsqueeze(1)
        h_omic_bag1 = torch.stack(h_omic1)  ### omic embeddings are stacked (to be used in co-attention)

        x_omic2 = [data[1]]  # needs to be wrapped in list (used to expect multiple omic sources)
        h_omic2 = [self.sig_networks2[idx].forward(sig_feat) for idx, sig_feat in
                  enumerate(x_omic2)]  ### each omic signature goes through it's own FC layer
        h_omic_bag2 = torch.stack(h_omic2)

        #x_path = data[1]
        #x_path = einops.rearrange(x_path, "b dim patches -> b patches dim")
       # h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag2, h_omic_bag1, h_omic_bag1)

        ### Path
        #h_path_trans = self.path_transformer(h_path_coattn)
        #A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        #A_path = torch.transpose(A_path, 1, 0)
        #A_path = A_path.squeeze(-1)
        #h_path = h_path.squeeze(0)
        ## h_path = torch.mm(F.softmax(A_path.t(), dim=1) , h_path) # keep batch dimension
        #h_path = self.path_rho(h_path).squeeze()

        ### Omic 1
        h_omic_trans1 = self.omic_transformer(h_path_coattn)
        A_omic1, h_omic1 = self.omic_attention_head(h_omic_trans1.squeeze(1))
        A_omic1 = torch.transpose(A_omic1, 1, 0)
        A_omic1 = A_omic1.squeeze(-1)
        h_omic1 = h_omic1.squeeze(0)
        # h_omic = torch.mm(F.softmax(A_omic.t(), dim=1), h_omic)
        h_omic1 = self.omic_rho(h_omic1).squeeze()

        ### Omic 2
        h_omic_trans2 = self.omic_transformer(h_omic_bag2)
        A_omic2, h_omic2 = self.omic_attention_head(h_omic_trans2.squeeze(1))
        A_omic2 = torch.transpose(A_omic2, 1, 0)
        A_omic = A_omic2.squeeze(-1)
        h_omic2 = h_omic2.squeeze(0)
        # h_omic = torch.mm(F.softmax(A_omic.t(), dim=1), h_omic)
        h_omic2 = self.omic_rho(h_omic2).squeeze()

        if self.fusion == 'bilinear':
            h = self.mm(h_omic1.unsqueeze(dim=0), h_omic2.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            if len(h_omic2.shape) == 1: # if batch size is 1
                h_omic2 = h_omic2.unsqueeze(dim=0)
                h_omic1 = h_omic1.unsqueeze(dim=0)
            h = self.mm(torch.cat([h_omic1, h_omic2], axis=1))

        ### Survival Layer
        logit = self.classifier(h)

        # happens in train loop
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        #
        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        if infer:
            return logit
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        loss = torch.mean(criterion(logit, labels))
        return loss, logit


    def captum(self, x_path, x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6):
        #x_path = torch.randn((10, 500, 1024))
        #x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6 = [torch.randn(10, size) for size in omic_sizes]
        x_omic = [x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6]
        h_path_bag = self.wsi_net(x_path)#.unsqueeze(1) ### path embeddings are fed through a FC layer
        h_path_bag = torch.reshape(h_path_bag, (500, 10, 256))
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path_trans = torch.reshape(h_path_trans, (10, 6, 256))
        A_path, h_path = self.path_attention_head(h_path_trans)
        A_path = F.softmax(A_path.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_path = torch.bmm(A_path, h_path).squeeze(dim=1)

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        h_omic_trans = torch.reshape(h_omic_trans, (10, 6, 256))
        A_omic, h_omic = self.omic_attention_head(h_omic_trans)
        A_omic = F.softmax(A_omic.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_omic = torch.bmm(A_omic, h_omic).squeeze(dim=1)

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=1))

        logits  = self.classifier(h)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        return risk

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


