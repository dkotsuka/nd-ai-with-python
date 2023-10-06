import os
import glob
import torch

from build_classifier import build_classifier
from get_pretrained_model import get_pretrained_model


def load_checkpoint(filepath, device='cpu'):
    """ Function to load a checkpoint.

    Args:
        filepath: path to the checkpoint file

    Returns:
        model: a PyTorch model
    """
    files = glob.glob(os.path.join(filepath, '*.pth'))

    if files:
        latest_file = max(files, key=os.path.getctime)
        filepath = latest_file
    else:
        print("No trainning checkpoint found.")
        exit()

    checkpoint = torch.load(filepath)

    model = build_classifier(
        checkpoint['input_size'], checkpoint['hidden_layers'], checkpoint['output_size'])
    model.load_state_dict(checkpoint['state_dict'])

    arch = checkpoint['arch']

    pretrained = get_pretrained_model(arch, device)

    return model, pretrained
