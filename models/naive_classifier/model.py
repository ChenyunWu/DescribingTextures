import torch.nn as nn

from models.layers.img_encoder import ResnetEncoder


class PhraseClassifier(nn.Module):
    def __init__(self, class_num, backbone='resnet101', pretrained_backbone=True, use_feats=(4,),
                 fc_dims=(512,)):
        super(PhraseClassifier, self).__init__()
        self.img_encoder = ResnetEncoder(backbone, pretrained_backbone, use_feats)

        in_dim = self.img_encoder.out_dim
        fc_layers = []
        if len(fc_dims) > 0:
            for fc_i, fc_dim in enumerate(fc_dims):
                fc_layer = nn.Sequential(nn.Linear(in_dim, fc_dim),
                                         nn.BatchNorm1d(fc_dim),
                                         nn.ReLU())
                fc_layers.append(fc_layer)
                in_dim = fc_dim
        fc_layers.append(nn.Linear(in_dim, class_num))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        img_feats = self.img_encoder(x)
        pred_scores = self.fc_layers(img_feats)
        return pred_scores
