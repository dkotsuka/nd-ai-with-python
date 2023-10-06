import torchvision.models as models


def get_pretrained_model(arch, device):
    """ Function to get and return a pretrained model.

    Returns:
        pretrained - pretrained model

    """
    model = None
    if (arch == 'vgg11'):
        model = models.vgg11
    elif (arch == 'vgg13'):
        model = models.vgg13
    elif (arch == 'vgg16'):
        model = models.vgg16
    elif (arch == 'vgg19'):
        model = models.vgg19
    elif (arch == 'alexnet'):
        model = models.alexnet
    elif (arch == 'resnet18'):
        model = models.resnet18
    elif (arch == 'resnet34'):
        model = models.resnet34
    else:
        print("Error: architecture not found")
        exit()

    pretrained = model(pretrained=True)
    pretrained = pretrained.to(device)
    for param in pretrained.parameters():
        param.requires_grad = False

    return pretrained
