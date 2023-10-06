import torch

from get_predict_input_args import get_predict_input_args
from load_checkpoint import load_checkpoint


def get_prediction(image, device):

    args = get_predict_input_args()
    classifier, pretrained = load_checkpoint(args.checkpoint_dir, device='cpu')

    classifier.to(device)
    image = image.unsqueeze(0)
    image = image.to(device)

    feature = pretrained(image)
    feature = feature.to(device)
    output = classifier.forward(feature)
    ps = torch.exp(output)

    return ps
