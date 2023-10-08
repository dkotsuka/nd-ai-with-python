from torch import nn


def build_classifier(ordered_dict):
    """ Build a classifier from an ordered dictionary of layers.

    Args:
        ordered_dict (OrderedDict): An ordered dictionary of layers.

    Returns:
        Sequential: A sequential container of layers.
    """

    return nn.Sequential(ordered_dict)
