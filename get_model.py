from torch import nn
import torchvision.models as models
from collections import OrderedDict

from build_classifier import build_classifier


def get_model(arch):
    classifier_archs = {
        "vgg": OrderedDict([('fc1', nn.Linear(25088, 4096)),
                            ('relu1', nn.ReLU()),
                            ('fc2', nn.Linear(4096, 512)),
                            ('relu2', nn.ReLU()),
                            ('fc3', nn.Linear(512, 256)),
                            ('relu3', nn.ReLU()),
                            ('fc4', nn.Linear(256, 102)),
                            ('softmax', nn.LogSoftmax(dim=1))]),
        "resnet": OrderedDict([('fc1', nn.Linear(2048, 1024)),
                               ('relu1', nn.ReLU()),
                               ('fc2', nn.Linear(1024, 512)),
                               ('relu2', nn.ReLU()),
                               ('fc3', nn.Linear(512, 256)),
                               ('relu3', nn.ReLU()),
                               ('fc4', nn.Linear(256, 102)),
                               ('softmax', nn.LogSoftmax(dim=1))])
    }

    if (arch == "vgg"):
        model = models.vgg16(pretrained=True)
    elif (arch == "resnet"):
        model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier_layer = build_classifier(classifier_archs[arch])

    if (arch == "vgg"):
        model.classifier = classifier_layer
    elif (arch == "resnet"):
        model.fc = classifier_layer

    return model
