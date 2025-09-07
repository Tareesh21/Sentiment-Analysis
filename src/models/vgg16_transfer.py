import torch.nn as nn
import torchvision.models as models

def get_vgg16(num_classes=7, pretrained=True):
    vgg = models.vgg16(pretrained=pretrained)
    vgg.classifier[6] = nn.Linear(4096, num_classes)
    return vgg
