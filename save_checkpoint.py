import torch
import datetime
import os

from torch import nn


def save_checkpoint(model, save_dir, arch, class_to_idx, learning_rate, optimizer, epoch_counter):
    """
    Function to save the trainning checkpoint.

    Returns:
        None
    """

    if (arch == "vgg"):
        classifier = model.classifier
    elif (arch == "resnet"):
        classifier = model.fc

    hidden_layers = []
    for index, layer in enumerate(classifier.children()):
        if isinstance(layer, nn.Linear):
            hidden_layers.append(layer.out_features)
    hidden_layers.pop()

    checkpoint = {'arch': arch,
                  'classifier_state_dict': classifier.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'learning_rate': learning_rate,
                  'class_to_idx': class_to_idx,
                  'epochs': epoch_counter
                  }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint, 'checkpoint-{}.pth'.format(arch))

    print("Checkpoint saved at: {}".format(datetime.datetime.now()))
