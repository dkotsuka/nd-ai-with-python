import os
import glob
import torch
from torch import optim

from get_model import get_model


def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    arch = checkpoint['arch']

    model = get_model(checkpoint['arch'])

    if (arch == 'vgg'):
        parameters = model.classifier.parameters()
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    elif (arch == 'resnet'):
        parameters = model.fc.parameters()
        model.fc.load_state_dict(checkpoint['classifier_state_dict'])

    optimizer = optim.SGD(parameters, checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    epoch_counter = checkpoint['epochs']

    return model, optimizer, epoch_counter
