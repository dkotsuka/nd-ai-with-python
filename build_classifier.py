from torch import nn


def build_classifier(input_size, hidden_sizes, output_size):
    """ Function to build a classifier model.

    Returns:
        model - classifier model
    """

    layers = []

    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())

    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    layers.append(nn.LogSoftmax(dim=1))

    model = nn.Sequential(*layers)

    return model
