import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck, model_urls


class ResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        # TODO: not sure if feat_dims are correct
        self.feat_dims = np.array([64, 256, 512, 1024, 2048])

    def extract_feats(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = self.maxpool(x)

        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        # f4_ = self.avgpool(f4)
        # f4_ = torch.flatten(f4_, 1)
        # pred = self.fc(f4_)

        return [f0, f1, f2, f3, f4]

    def extract_flat_feats(self, x):
        feats = self.extract_feats(x)
        flat_feats = []
        for fi, f in enumerate(feats):
            if fi <= 4:
                ff = self.avgpool(f)
                ff = torch.flatten(ff, 1)
                flat_feats.append(ff)
        return flat_feats


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


class ResnetEncoder(nn.Module):
    def __init__(self, backbone='resnet101', pretrained_backbone=True, use_feats=(4,)):
        super(ResnetEncoder, self).__init__()

        self.use_feats = use_feats

        if backbone == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained_backbone)
        else:
            raise NotImplementedError

        # self.fc_layers = None
        feats_dim = 0
        for fi in use_feats:
            feats_dim += self.backbone.feat_dims[fi]
        self.out_dim = feats_dim
        return

    def forward(self, x):
        feats = self.backbone.extract_flat_feats(x)
        feat = torch.cat([feats[fi] for fi in self.use_feats], dim=1)
        return feat


def build_transforms(is_train=True):
    # follow the transforms in `pytorch/examples/imagenet`:line 202

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
