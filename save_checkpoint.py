import torch
import datetime
import os

from torch import nn


def save_checkpoint(model, save_dir, arch):
    """
    Function to save the trainning checkpoint.

    Returns:
        None
    """

    hidden_layers = []
    for index, layer in enumerate(model.children()):
        if isinstance(layer, nn.Linear):
            hidden_layers.append(layer.out_features)
    hidden_layers.pop()

    checkpoint = {'input_size': 1000,
                  'output_size': 102,
                  'hidden_layers': hidden_layers,
                  'state_dict': model.state_dict(),
                  'arch': arch}

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint, save_dir + '/checkpoint' + date_string + '.pth')
